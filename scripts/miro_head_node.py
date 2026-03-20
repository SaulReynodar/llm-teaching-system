#!/usr/bin/env python3
import rospy
import time
import threading
import subprocess
import os
import re
import random

from std_msgs.msg import String, Float32MultiArray, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


KIN_TOPIC = "/miro/control/kinematic_joints"
COS_TOPIC = "/miro/control/cosmetic_joints"
CMD_VEL_TOPIC = "/miro/control/cmd_vel"


def norm_text(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class MiroBehaviourNode:
    def __init__(self):
        rospy.init_node("miro_behaviour")

        self.pub_kin = rospy.Publisher(KIN_TOPIC, JointState, queue_size=1)
        self.pub_cos = rospy.Publisher(COS_TOPIC, Float32MultiArray, queue_size=1)
        self.pub_cmd = rospy.Publisher(CMD_VEL_TOPIC, TwistStamped, queue_size=1)

        self.pub_answer = rospy.Publisher("/miro/answer", String, queue_size=10)
        self.pub_speaking = rospy.Publisher("/miro/speaking", Bool, queue_size=10, latch=True)

        rospy.Subscriber("/teacher/state", String, self.on_teacher_state)
        rospy.Subscriber("/teacher/question_to_miro", String, self.on_teacher_question)
        rospy.Subscriber("/teacher/last_chunk", String, self.on_last_chunk)
        rospy.Subscriber("/teacher/story_summary", String, self.on_story_summary)

        self.teacher_state = "IDLE"
        self.last_chunk_text = ""
        self.story_summary = ""

        self.miro_correct_probability = float(rospy.get_param("~miro_correct_probability", 0.6))
        self.llm_model = rospy.get_param("~llm_model", "gpt-4o-mini")

        # ---------------- HEAD POSITIONS ----------------

        self.head_normal = 0.5
        self.head_up_small = 0.2
        self.head_down = 0.8
        self.head_up_full = 0.0

        # ---------------- BLINK ----------------

        self.blink_interval = 2.0
        self.blink_close_time = 0.12

        # ---------------- CIRCLE ----------------

        self.circle_linear = 0.60
        self.circle_angular = 15.0
        self.circle_seconds = 1.6

        # ---------------- PIPER ----------------

        self.piper_bin = "/home/saul_reynodar/piper/piper"
        self.piper_model = "/home/saul_reynodar/piper/malechild.onnx"
        self.wav_path = "/tmp/miro.wav"

        self.motion_lock = threading.Lock()

        # cosmetic mapping
        # [tail_ud, tail_lr, eye_l, eye_r, ear_l, ear_r]
        self.tail_left = 0.0
        self.tail_right = 1.0
        self.ears_up = 1.0
        self.ears_down = 0.0

        self.cosmetic = [0.5, 0.5, 0, 0, 1, 1]
        self.last_teacher_voice = time.time()

        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if OpenAI is not None and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                rospy.loginfo("MiRo LLM client ready.")
            except Exception as e:
                rospy.logwarn("MiRo LLM init failed: %s", str(e))
        else:
            rospy.logwarn("MiRo LLM not active. Fallback answers will be used.")

        threading.Thread(target=self.blink_loop, daemon=True).start()
        threading.Thread(target=self.tail_wag_loop, daemon=True).start()
        threading.Thread(target=self.ear_idle_monitor, daemon=True).start()

        rospy.loginfo("MiRo behaviour ready")

    # ---------------------------------------------------
    # LLM helpers
    # ---------------------------------------------------

    def llm_text(self, system_prompt, user_prompt):
        if self.client is None:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            rospy.logwarn("MiRo LLM request failed: %s", str(e))
            return ""

    def generate_miro_doubt_detail(self):
        system_prompt = (
            "You are MiRo, a curious child robot in class. "
            "Ask one short, natural doubt based on the story context. "
            "Use one sentence only."
        )
        user_prompt = (
            "Story summary: {summary}\n"
            "Recent part: {chunk}\n"
            "Generate one simple doubt."
        ).format(summary=self.story_summary, chunk=self.last_chunk_text)

        text = self.llm_text(system_prompt, user_prompt)
        return text.strip()

    def generate_miro_answer(self, question, correct_answer, should_be_correct):
        if should_be_correct:
            system_prompt = (
                "You are MiRo, a child robot student. "
                "Answer the question correctly in one short sentence. "
                "Keep it natural and simple."
            )
            user_prompt = (
                "Story summary: {summary}\n"
                "Question: {q}\n"
                "Correct answer meaning: {a}"
            ).format(summary=self.story_summary, q=question, a=correct_answer)

            text = self.llm_text(system_prompt, user_prompt)
            return text.strip() or correct_answer

        system_prompt = (
            "You are MiRo, a child robot student. "
            "Give a short but wrong answer to the question. "
            "It should sound related to the story, but still be incorrect. "
            "Do not say you do not know."
        )
        user_prompt = (
            "Story summary: {summary}\n"
            "Question: {q}\n"
            "Correct answer: {a}"
        ).format(summary=self.story_summary, q=question, a=correct_answer)

        text = self.llm_text(system_prompt, user_prompt)
        if text:
            return text.strip()

        return "I think it was because he was the strongest knight."

    # ---------------------------------------------------
    # Cosmetics
    # ---------------------------------------------------

    def publish_cosmetics(self):
        msg = Float32MultiArray()
        msg.data = self.cosmetic
        self.pub_cos.publish(msg)

    # ---------------------------------------------------
    # Subscribers
    # ---------------------------------------------------

    def on_last_chunk(self, msg):
        self.last_chunk_text = (msg.data or "").strip()

    def on_story_summary(self, msg):
        self.story_summary = (msg.data or "").strip()

    # ---------------------------------------------------
    # Tail wag
    # ---------------------------------------------------

    def tail_wag_loop(self):
        while not rospy.is_shutdown():
            self.cosmetic[1] = self.tail_right
            self.publish_cosmetics()
            time.sleep(0.25)

            self.cosmetic[1] = self.tail_left
            self.publish_cosmetics()
            time.sleep(0.25)

    # ---------------------------------------------------
    # Teacher state
    # ---------------------------------------------------

    def on_teacher_state(self, msg):
        state = (msg.data or "").strip()
        self.teacher_state = state

        if state == "SPEAKING":
            self.last_teacher_voice = time.time()
            self.cosmetic[4] = self.ears_up
            self.cosmetic[5] = self.ears_up
            self.publish_cosmetics()

        if state == "MIRO_CORRECT":
            threading.Thread(target=self.joy_circle_360, daemon=True).start()

        if state == "MIRO_WRONG":
            threading.Thread(target=self.guilt_reaction, daemon=True).start()

        if state == "MIRO_OKAY":
            self.send_head(self.head_normal)

    # ---------------------------------------------------
    # Long silence detection for ears
    # ---------------------------------------------------

    def ear_idle_monitor(self):
        while not rospy.is_shutdown():
            if time.time() - self.last_teacher_voice > 3.0:
                self.cosmetic[4] = self.ears_down
                self.cosmetic[5] = self.ears_down
                self.publish_cosmetics()

            time.sleep(0.5)

    # ---------------------------------------------------
    # Teacher questions / prompts
    # ---------------------------------------------------

    def on_teacher_question(self, msg):
        payload = (msg.data or "").strip()

        if payload == "__MIRO_INTERRUPT__":
            self.pitch_once()
            self.send_head(self.head_up_full)

            text = "Excuse me teacher, I have a doubt."
            self.pub_answer.publish(text)
            self.miro_speak(text)

            self.send_head(self.head_normal)
            self.left_right_shake()
            return

        if payload == "__MIRO_DOUBT_DETAIL__":
            self.send_head(self.head_up_full)

            text = self.generate_miro_doubt_detail()
            if not text:
                text = "Who is Merlin?"

            self.pub_answer.publish(text)
            self.miro_speak(text)

            self.send_head(self.head_normal)
            self.left_right_shake()
            return

        if payload.startswith("MID::"):
            parts = payload.split("::", 2)
            question = parts[1].strip() if len(parts) > 1 else ""
            correct = parts[2].strip() if len(parts) > 2 else ""

            should_be_correct = (random.random() < self.miro_correct_probability)
            ans = self.generate_miro_answer(question, correct, should_be_correct)

            self.send_head(self.head_up_full)
            self.pub_answer.publish(ans)
            self.miro_speak(ans)
            self.send_head(self.head_normal)
            return

        if payload.startswith("QUIZ::"):
            parts = payload.split("::", 3)

            question = parts[2].strip() if len(parts) > 2 else ""
            correct = parts[3].strip() if len(parts) > 3 else ""

            should_be_correct = (random.random() < self.miro_correct_probability)
            ans = self.generate_miro_answer(question, correct, should_be_correct)

            self.send_head(self.head_up_full)
            self.pub_answer.publish(ans)
            self.miro_speak(ans)
            self.send_head(self.head_normal)
            return

    # ---------------------------------------------------
    # Head control
    # ---------------------------------------------------

    def send_head(self, yaw=0.5, pitch=0, roll=0):
        js = JointState()
        js.name = ["lift", "yaw", "pitch", "roll"]
        js.position = [0.35, yaw, pitch, roll]
        self.pub_kin.publish(js)

    def nod(self):
        with self.motion_lock:
            self.send_head(self.head_up_small)
            time.sleep(0.12)

            self.send_head(self.head_down)
            time.sleep(0.12)

            self.send_head(self.head_normal)

    def pitch_once(self):
        with self.motion_lock:
            self.send_head(self.head_normal, pitch=0.4)
            time.sleep(0.1)

            self.send_head(self.head_normal, pitch=-0.4)
            time.sleep(0.1)

            self.send_head(self.head_normal, pitch=0)

    def left_right_shake(self):
        with self.motion_lock:
            self.send_head(self.head_normal, roll=-0.4)
            time.sleep(0.12)

            self.send_head(self.head_normal, roll=0.4)
            time.sleep(0.12)

            self.send_head(self.head_normal)

    def guilt_reaction(self):
        with self.motion_lock:
            self.send_head(self.head_down, roll=0.6)
            time.sleep(1.0)
            self.send_head(self.head_normal)

    # ---------------------------------------------------
    # Circle
    # ---------------------------------------------------

    def joy_circle_360(self):
        msg = TwistStamped()
        msg.twist.linear.x = self.circle_linear
        msg.twist.angular.z = self.circle_angular

        rate = rospy.Rate(20)
        start = time.time()

        while not rospy.is_shutdown() and (time.time() - start) < self.circle_seconds:
            msg.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(msg)
            rate.sleep()

        stop = TwistStamped()
        stop.twist.linear.x = 0
        stop.twist.angular.z = 0

        for _ in range(6):
            stop.header.stamp = rospy.Time.now()
            self.pub_cmd.publish(stop)
            time.sleep(0.02)

    # ---------------------------------------------------
    # Blink
    # ---------------------------------------------------

    def blink_loop(self):
        time.sleep(1)

        while not rospy.is_shutdown():
            self.cosmetic[2] = 1
            self.cosmetic[3] = 1
            self.publish_cosmetics()

            time.sleep(self.blink_close_time)

            self.cosmetic[2] = 0
            self.cosmetic[3] = 0
            self.publish_cosmetics()

            time.sleep(self.blink_interval)

    # ---------------------------------------------------
    # Speak
    # ---------------------------------------------------

    def miro_speak(self, text):
        rospy.loginfo("[MIRO] %s", text)

        self.pub_speaking.publish(True)

        self.cosmetic[4] = self.ears_down
        self.cosmetic[5] = self.ears_down
        self.publish_cosmetics()

        with self.motion_lock:
            subprocess.run(
                [self.piper_bin, "--model", self.piper_model, "--output_file", self.wav_path],
                input=text.encode()
            )

            subprocess.call(["aplay", self.wav_path])

        self.cosmetic[4] = self.ears_up
        self.cosmetic[5] = self.ears_up
        self.publish_cosmetics()

        self.pub_speaking.publish(False)

    # ---------------------------------------------------
    # Main
    # ---------------------------------------------------

    def spin(self):
        rate = rospy.Rate(20)
        prev = "IDLE"

        while not rospy.is_shutdown():
            if prev == "SPEAKING" and self.teacher_state == "IDLE":
                threading.Thread(target=self.nod, daemon=True).start()

            prev = self.teacher_state
            rate.sleep()


if __name__ == "__main__":
    node = MiroBehaviourNode()
    node.spin()
