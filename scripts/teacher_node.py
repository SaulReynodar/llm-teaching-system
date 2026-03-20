#!/usr/bin/env python3
import os
import time
import json
import threading
import subprocess
import signal

import rospy
from std_msgs.msg import String, Bool

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class TeacherNode:
    def __init__(self):
        rospy.init_node("teacher_node")

        self.topic_name = rospy.get_param("~topic_name", "King Arthur")

        # Human timing
        self.wait_human_seconds = float(rospy.get_param("~wait_seconds", 5.0))          # after "What is your doubt?"
        self.human_answer_timeout = float(rospy.get_param("~human_answer_timeout", 10.0))  # question answer window
        self.miro_answer_timeout = float(rospy.get_param("~miro_answer_timeout", 10.0))

        self.quiz_count = int(rospy.get_param("~quiz_count", 5))
        self.sentence_pause = float(rospy.get_param("~sentence_pause", 0.25))
        self.miro_delay_lines_after_human = int(rospy.get_param("~miro_delay_lines_after_human", 3))

        self.piper_bin = rospy.get_param("~piper_bin", "/home/saul_reynodar/piper/piper")
        self.piper_model = rospy.get_param("~piper_model", "/home/saul_reynodar/piper/en_US-libritts-high.onnx")
        self.wav_path = rospy.get_param("~wav_path", "/tmp/teacher.wav")

        self.llm_model = rospy.get_param("~llm_model", "gpt-4o-mini")
        self.answer_max_sentences = int(rospy.get_param("~answer_max_sentences", 3))

        self.pub_text = rospy.Publisher("/teacher/text", String, queue_size=10)
        self.pub_state = rospy.Publisher("/teacher/state", String, queue_size=10, latch=True)
        self.pub_phase = rospy.Publisher("/teacher/phase", String, queue_size=10, latch=True)
        self.pub_allow_interrupt = rospy.Publisher("/teacher/allow_miro_interrupt", Bool, queue_size=10, latch=True)
        self.pub_last_chunk = rospy.Publisher("/teacher/last_chunk", String, queue_size=10)
        self.pub_question_to_miro = rospy.Publisher("/teacher/question_to_miro", String, queue_size=10)
        self.pub_story_summary = rospy.Publisher("/teacher/story_summary", String, queue_size=10, latch=True)

        rospy.Subscriber("/human/text", String, self.on_human_text)
        rospy.Subscriber("/miro/speaking", Bool, self.on_miro_speaking)
        rospy.Subscriber("/miro/answer", String, self.on_miro_answer)

        self._lock = threading.Lock()
        self._human_event = threading.Event()
        self._human_last = ""
        self._human_last_time = 0.0

        self._miro_answer_event = threading.Event()
        self._miro_answer = ""

        self._miro_speaking = False
        self._miro_speaking_event = threading.Event()

        self._play_proc = None
        self._interrupt_flag = threading.Event()

        self.last_chunk_text = ""
        self.story_lines_cache = []
        self.story_summary_cache = ""
        self.mid_question_cache = ""
        self.mid_answer_cache = ""

        self.human_interrupted_recently = False
        self.lines_since_last_human_interrupt = 999
        self.miro_doubt_done = False

        self.client = None
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if OpenAI is not None and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                rospy.loginfo("Teacher LLM client ready.")
            except Exception as e:
                rospy.logwarn("Teacher LLM init failed: %s", str(e))
        else:
            rospy.logwarn("Teacher LLM not active. Using fallback logic if needed.")

        rospy.loginfo("Teacher ready (human doubt priority + answer retry).")

    # ---------------- basic pub helpers ----------------

    def set_state(self, s):
        self.pub_state.publish(String(data=s))

    def pulse_state(self, s, pulse_sec=0.10):
        self.pub_state.publish(String(data=s))
        time.sleep(pulse_sec)
        self.pub_state.publish(String(data="IDLE"))

    def set_phase(self, p):
        self.pub_phase.publish(String(data=p))

    def allow_miro_interrupt(self, on):
        self.pub_allow_interrupt.publish(Bool(data=on))

    def say(self, text):
        rospy.loginfo("\n[TEACHER SAYS] %s\n", text)
        self.pub_text.publish(String(data=text))

    # ---------------- subscribers ----------------

    def on_miro_speaking(self, msg):
        self._miro_speaking = bool(msg.data)
        self._miro_speaking_event.set()

    def on_miro_answer(self, msg):
        txt = (msg.data or "").strip()
        if not txt:
            return
        self._miro_answer = txt
        self._miro_answer_event.set()

    def on_human_text(self, msg):
        txt = (msg.data or "").strip()
        if not txt or len(txt) < 2:
            return
        with self._lock:
            self._human_last = txt
            self._human_last_time = time.time()
            self._human_event.set()

        self._interrupt_flag.set()
        self.human_interrupted_recently = True
        self.lines_since_last_human_interrupt = 0
        rospy.loginfo("[HUMAN SAID] %s", txt)

    # ---------------- timing coordination ----------------

    def wait_until_miro_silent(self, timeout=20.0):
        start = time.time()
        while self._miro_speaking and (time.time() - start) < timeout and not rospy.is_shutdown():
            time.sleep(0.02)

    def wait_for_miro_turn(self, timeout_start=2.0, timeout_end=25.0):
        start = time.time()
        while not rospy.is_shutdown() and (time.time() - start) < timeout_start:
            if self._miro_speaking:
                break
            self._miro_speaking_event.clear()
            self._miro_speaking_event.wait(timeout=0.05)

        start2 = time.time()
        while not rospy.is_shutdown() and (time.time() - start2) < timeout_end:
            if not self._miro_speaking:
                return
            self._miro_speaking_event.clear()
            self._miro_speaking_event.wait(timeout=0.05)

    # ---------------- piper ----------------

    def _tts_generate_wav(self, text):
        if not os.path.exists(self.piper_bin) or not os.path.exists(self.piper_model):
            rospy.logwarn("Teacher: Piper missing. bin=%s model=%s", self.piper_bin, self.piper_model)
            return False
        cmd = [self.piper_bin, "--model", self.piper_model, "--output_file", self.wav_path]
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
        return True

    def _stop_playback(self):
        try:
            if self._play_proc and self._play_proc.poll() is None:
                self._play_proc.send_signal(signal.SIGINT)
                time.sleep(0.05)
                if self._play_proc.poll() is None:
                    self._play_proc.terminate()
                time.sleep(0.05)
                if self._play_proc.poll() is None:
                    self._play_proc.kill()
        except Exception:
            pass
        self._play_proc = None

    def speak_interruptible(self, text):
        self.wait_until_miro_silent()

        self._interrupt_flag.clear()
        self.set_state("SPEAKING")
        self.say(text)

        try:
            ok = self._tts_generate_wav(text)
        except Exception as e:
            rospy.logwarn("Teacher TTS generate failed: %s", str(e))
            ok = False

        if not ok:
            self.set_state("IDLE")
            return True

        try:
            self._play_proc = subprocess.Popen(
                ["aplay", self.wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while self._play_proc.poll() is None and not rospy.is_shutdown():
                if self._interrupt_flag.is_set():
                    self._stop_playback()
                    self.set_state("IDLE")
                    return False
                time.sleep(0.02)
        except Exception as e:
            rospy.logwarn("Teacher playback failed: %s", str(e))

        self.set_state("IDLE")
        return True

    # ---------------- human wait helpers ----------------

    def _consume_latest_human(self):
        with self._lock:
            return self._human_last

    def wait_for_human(self, seconds):
        self.set_phase("WAITING_HUMAN")
        self._human_event.clear()
        got = self._human_event.wait(timeout=seconds)
        self.set_phase("IDLE")
        if not got:
            return ""
        return self._consume_latest_human()

    def wait_for_new_human(self, seconds, after_time):
        self.set_phase("WAITING_HUMAN")
        end_time = time.time() + seconds

        while time.time() < end_time and not rospy.is_shutdown():
            with self._lock:
                if self._human_last_time > after_time and self._human_last.strip():
                    self.set_phase("IDLE")
                    return self._human_last.strip()
            time.sleep(0.05)

        self.set_phase("IDLE")
        return ""

    # ---------------- LLM helpers ----------------

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
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            rospy.logwarn("Teacher LLM request failed: %s", str(e))
            return ""

    def offline_story_bundle(self):
        story_lines = [
            "In a magical land, a young boy named Arthur dreamed of being brave and kind.",
            "Arthur lived quietly, but he always helped people when they were scared or sad.",
            "One day, a bright crowd gathered near a stone in the town square.",
            "Inside the stone, there was a sword that shined like silver lightning.",
            "People whispered that only the true king could pull this sword out of the stone.",
            "Strong knights tried again and again, but the sword did not move even a little.",
            "Arthur walked closer, not to show off, but only to see what was happening.",
            "Merlin the wizard watched Arthur carefully, as if he knew a secret.",
            "Arthur held the sword gently, and the sword came out smoothly, like it was waiting for him.",
            "The crowd gasped, because they understood something important had happened.",
            "Merlin said Arthur was chosen to lead with courage and a good heart.",
            "Arthur did not become proud; he promised to protect people who cannot protect themselves.",
            "Later, Arthur created the Round Table so no knight would feel less important.",
            "At the Round Table, everyone spoke politely and listened to each other.",
            "When danger came, Arthur lifted Excalibur and gave people hope.",
            "That is why King Arthur is remembered as fair, brave, and kind."
        ]

        mid_question = "What proved that Arthur was chosen to be the true king?"
        mid_answer = "He pulled the sword out of the stone when no one else could."

        quiz = [
            ("Where was the sword placed?", "Inside a stone in the town square."),
            ("Who was watching Arthur carefully?", "Merlin the wizard."),
            ("Why did Arthur make the Round Table?", "So everyone would be equal and respected."),
            ("What did Arthur promise to do?", "To protect people who cannot protect themselves."),
            ("What do people remember Arthur for?", "Being fair, brave, and kind."),
        ]

        summary = (
            "Arthur is a kind young boy. A sword is stuck in a stone. Many knights fail to pull it out. "
            "Merlin watches Arthur. Arthur pulls out the sword and is chosen as king. "
            "He protects people, creates the Round Table, and is remembered as fair, brave, and kind."
        )
        return story_lines, mid_question, mid_answer, quiz, summary

    def build_story_bundle_with_llm(self):
        system_prompt = (
            "You are a classroom storytelling assistant. "
            "Return valid JSON only with keys: story_lines, mid_question, mid_answer, quiz, summary. "
            "story_lines must be exactly 16 short child-friendly sentences. "
            "quiz must contain exactly 5 items, each as an object with keys q and a. "
            "Keep the facts around King Arthur, sword in the stone, Merlin, bravery, fairness, and Round Table."
        )
        user_prompt = (
            "Create a simple classroom story about King Arthur. "
            "Make it good for spoken teaching and question answering."
        )

        raw = self.llm_text(system_prompt, user_prompt)
        if not raw:
            return None

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
            story_lines = data.get("story_lines", [])
            mid_question = (data.get("mid_question") or "").strip()
            mid_answer = (data.get("mid_answer") or "").strip()
            summary = (data.get("summary") or "").strip()

            quiz = []
            for item in data.get("quiz", []):
                q = (item.get("q") or "").strip()
                a = (item.get("a") or "").strip()
                if q and a:
                    quiz.append((q, a))

            if len(story_lines) != 16 or not mid_question or not mid_answer or len(quiz) != 5:
                return None

            story_lines = [str(x).strip() for x in story_lines]
            return story_lines, mid_question, mid_answer, quiz, summary
        except Exception as e:
            rospy.logwarn("Teacher LLM story parse failed: %s", str(e))
            return None

    def build_story(self):
        bundle = self.build_story_bundle_with_llm()
        if bundle is None:
            rospy.loginfo("Using offline story fallback.")
            bundle = self.offline_story_bundle()
        else:
            rospy.loginfo("Using LLM-generated story bundle.")

        story_lines, mid_q, mid_a, quiz, summary = bundle

        self.story_lines_cache = story_lines[:]
        self.story_summary_cache = summary
        self.mid_question_cache = mid_q
        self.mid_answer_cache = mid_a

        self.pub_story_summary.publish(String(data=summary))
        return story_lines, mid_q, mid_a, quiz

    def answer_doubt_with_llm(self, doubt_text, speaker_type="human"):
        context_story = " ".join(self.story_lines_cache) if self.story_lines_cache else self.story_summary_cache

        system_prompt = (
            "You are a warm primary-school teacher. "
            "Answer the doubt correctly using the story context. "
            "Use simple English. Keep it short and clear for speech. "
            "Do not mention being an AI."
        )

        user_prompt = (
            "Topic: {topic}\n"
            "Story summary: {summary}\n"
            "Recent story part: {last_chunk}\n"
            "Full story: {full_story}\n"
            "{speaker_type} doubt: {doubt}\n"
            "Answer in at most {n} short sentences."
        ).format(
            topic=self.topic_name,
            summary=self.story_summary_cache,
            last_chunk=self.last_chunk_text,
            full_story=context_story,
            speaker_type=speaker_type,
            doubt=doubt_text,
            n=self.answer_max_sentences,
        )

        answer = self.llm_text(system_prompt, user_prompt)
        return answer.strip()

    def evaluate_answer_with_llm(self, question, correct_answer, given_answer):
        system_prompt = (
            "You are grading a short classroom answer. "
            "Return exactly one word only: CORRECT, WRONG, or UNCLEAR."
        )
        user_prompt = (
            "Question: {q}\n"
            "Expected answer: {a}\n"
            "Given answer: {g}\n"
            "Judge by meaning, not exact wording."
        ).format(q=question, a=correct_answer, g=given_answer)

        result = self.llm_text(system_prompt, user_prompt).upper().strip()
        if "CORRECT" in result:
            return "CORRECT"
        if "WRONG" in result:
            return "WRONG"
        return "UNCLEAR"

    # ---------------- speech classification ----------------

    def is_generic_doubt_trigger(self, text):
        t = (text or "").strip().lower()
        generic_doubt_phrases = [
            "i have a doubt",
            "i have doubt",
            "i have a question",
            "teacher i have a doubt",
            "i want to ask something",
        ]
        return any(t == p for p in generic_doubt_phrases)

    def seems_like_question(self, text):
        t = (text or "").strip().lower()
        starters = ["who", "what", "why", "how", "when", "where", "which", "is", "are", "can", "did", "does"]
        return ("?" in t) or any(t.startswith(s + " ") for s in starters)

    def is_human_idk(self, text):
        t = (text or "").strip().lower()
        return t in [
            "i don't know",
            "i dont know",
            "i do not know",
            "dont know",
            "don't know",
            "no idea",
            "i have no idea",
        ]

    def is_probably_unclear_text(self, text):
        t = (text or "").strip().lower()

        if len(t) < 4:
            return True
        if self.is_generic_doubt_trigger(t):
            return False
        if self.seems_like_question(t):
            return False
        if self.is_human_idk(t):
            return False

        word_count = len(t.split())
        if word_count <= 2:
            return True

        unclear_markers = ["uh", "um", "hmm", "mmm"]
        if t in unclear_markers:
            return True

        return False

    # ---------------- doubt handling ----------------

    def explain_miro_doubt(self, doubt_text):
        self.set_phase("DOUBT_EXPLAIN")
        answer = self.answer_doubt_with_llm(doubt_text, speaker_type="miro")
        if answer:
            self.speak_interruptible(answer)
        else:
            self.speak_interruptible("Let me explain simply.")
            if self.last_chunk_text:
                self.speak_interruptible(self.last_chunk_text)

    def explain_human_doubt(self, doubt_text):
        self.set_phase("HUMAN_DOUBT")
        self.allow_miro_interrupt(False)

        answer = self.answer_doubt_with_llm(doubt_text, speaker_type="human")
        if answer:
            self.speak_interruptible(answer)
        else:
            self.speak_interruptible("Let me explain simply.")
            if self.last_chunk_text:
                self.speak_interruptible(self.last_chunk_text)

    def ask_human_to_repeat_doubt_and_listen(self):
        self.speak_interruptible("I did not understand clearly. Can you tell me your doubt again?")
        prompt_time = time.time()
        actual_doubt = self.wait_for_new_human(self.wait_human_seconds, prompt_time)

        if actual_doubt:
            if self.is_generic_doubt_trigger(actual_doubt):
                self.speak_interruptible("What is your doubt?")
                prompt_time2 = time.time()
                actual_doubt_2 = self.wait_for_new_human(self.wait_human_seconds, prompt_time2)

                if actual_doubt_2 and self.seems_like_question(actual_doubt_2):
                    self.explain_human_doubt(actual_doubt_2)
                elif actual_doubt_2:
                    self.speak_interruptible("I still did not understand clearly. Please ask again later.")
                else:
                    self.speak_interruptible("I could not hear your doubt clearly. Please ask again later.")
                return

            if self.seems_like_question(actual_doubt):
                self.explain_human_doubt(actual_doubt)
                return

            self.speak_interruptible("I still did not understand clearly. Please ask again later.")
            return

        self.speak_interruptible("I could not hear your doubt clearly. Please ask again later.")

    def clarify_human_doubt(self, doubt_text):
        self.set_phase("HUMAN_DOUBT")
        self.allow_miro_interrupt(False)

        if self.is_generic_doubt_trigger(doubt_text):
            self.speak_interruptible("What is your doubt?")
            prompt_time = time.time()
            actual_doubt = self.wait_for_new_human(self.wait_human_seconds, prompt_time)

            if actual_doubt:
                if self.seems_like_question(actual_doubt):
                    self.explain_human_doubt(actual_doubt)
                else:
                    self.ask_human_to_repeat_doubt_and_listen()
            else:
                self.speak_interruptible("I could not hear your doubt clearly. Please ask again.")
            return

        if self.seems_like_question(doubt_text):
            self.explain_human_doubt(doubt_text)
            return

        self.ask_human_to_repeat_doubt_and_listen()

    def should_trigger_miro_doubt_now(self):
        if self.miro_doubt_done:
            return False
        if self.human_interrupted_recently and self.lines_since_last_human_interrupt < self.miro_delay_lines_after_human:
            return False
        return True

    def miro_doubt_sequence(self):
        self.set_phase("MIRO_DOUBT")
        self.allow_miro_interrupt(False)

        _ = self.ask_miro_and_wait("__MIRO_INTERRUPT__")
        self.speak_interruptible("Okay MiRo. What is your doubt?")
        doubt_text = self.ask_miro_and_wait("__MIRO_DOUBT_DETAIL__")

        self.speak_interruptible("Okay MiRo. I understand your doubt.")
        self.explain_miro_doubt(doubt_text)

        self.speak_interruptible("Anybody have doubts? You can ask now.")
        human = self.wait_for_human(self.wait_human_seconds)
        if human:
            self.clarify_human_doubt(human)

        self.miro_doubt_done = True

    # ---------------- answer feedback ----------------

    def feedback_for_human_answer(self, question, correct_answer, human_answer):
        result = self.evaluate_answer_with_llm(question, correct_answer, human_answer)

        if result == "CORRECT":
            self.speak_interruptible("Very good. That is the right answer.")
        elif result == "WRONG":
            self.speak_interruptible("That answer is not correct.")
            self.speak_interruptible("The correct answer is: " + correct_answer)
        else:
            self.speak_interruptible("I am not fully sure about that answer.")
            self.speak_interruptible("The expected answer is: " + correct_answer)

    def teacher_feedback_for_miro(self, question, miro_ans, correct_ans):
        if not miro_ans:
            self.set_phase("QUIZ_FEEDBACK")
            self.speak_interruptible("MiRo did not answer. I will tell you the right answer.")
            self.speak_interruptible(correct_ans)
            return

        low = miro_ans.strip().lower()

        if low in ("i don't know", "i dont know", "i do not know"):
            self.set_phase("QUIZ_FEEDBACK")
            self.speak_interruptible("It is okay MiRo.")
            self.pulse_state("MIRO_OKAY")
            self.speak_interruptible("The correct answer is:")
            self.speak_interruptible(correct_ans)
            return

        result = self.evaluate_answer_with_llm(question, correct_ans, miro_ans)

        if result == "CORRECT":
            self.set_phase("QUIZ_FEEDBACK")
            self.speak_interruptible("Very good MiRo! This is the right answer.")
            self.pulse_state("MIRO_CORRECT")
            return

        self.set_phase("QUIZ_FEEDBACK")
        self.speak_interruptible("That answer is wrong MiRo.")
        self.pulse_state("MIRO_WRONG")
        self.speak_interruptible("It is okay MiRo.")
        self.pulse_state("MIRO_OKAY")
        self.speak_interruptible("I will tell you the right answer.")
        self.speak_interruptible(correct_ans)

    # ---------------- miro ask/wait ----------------

    def ask_miro_and_wait(self, payload):
        self.set_phase("WAITING_MIRO")
        self._miro_answer = ""
        self._miro_answer_event.clear()
        self._miro_speaking_event.clear()

        self.pub_question_to_miro.publish(String(data=payload))
        got = self._miro_answer_event.wait(timeout=self.miro_answer_timeout)
        ans = self._miro_answer if got else ""
        self.wait_for_miro_turn()
        return ans

    # ---------------- shared human first logic ----------------

    def handle_question_with_human_priority(self, question, correct_answer, payload_prefix):
        self.speak_interruptible("Anybody want to answer or ask a doubt?")

        human = self.wait_for_human(self.human_answer_timeout)

        if human:
            if self.is_human_idk(human):
                miro_ans = self.ask_miro_and_wait(payload_prefix)
                self.teacher_feedback_for_miro(question, miro_ans, correct_answer)
                return

            if self.is_generic_doubt_trigger(human):
                self.clarify_human_doubt(human)
                return

            if self.seems_like_question(human):
                self.explain_human_doubt(human)
                return

            if self.is_probably_unclear_text(human):
                self.speak_interruptible("Tell me clearly.")
                second_try = self.wait_for_human(self.human_answer_timeout)

                if second_try:
                    if self.is_human_idk(second_try):
                        miro_ans = self.ask_miro_and_wait(payload_prefix)
                        self.teacher_feedback_for_miro(question, miro_ans, correct_answer)
                        return

                    if self.seems_like_question(second_try):
                        self.explain_human_doubt(second_try)
                        return

                    if self.is_probably_unclear_text(second_try):
                        self.speak_interruptible("I still did not understand clearly.")
                        miro_ans = self.ask_miro_and_wait(payload_prefix)
                        self.teacher_feedback_for_miro(question, miro_ans, correct_answer)
                        return

                    self.feedback_for_human_answer(question, correct_answer, second_try)
                    return

                miro_ans = self.ask_miro_and_wait(payload_prefix)
                self.teacher_feedback_for_miro(question, miro_ans, correct_answer)
                return

            self.feedback_for_human_answer(question, correct_answer, human)
            return

        miro_ans = self.ask_miro_and_wait(payload_prefix)
        self.teacher_feedback_for_miro(question, miro_ans, correct_answer)

    # ---------------- story helpers ----------------

    def deliver_story_line(self, line):
        self.last_chunk_text = line
        self.pub_last_chunk.publish(String(data=line))

        finished = self.speak_interruptible(line)
        time.sleep(self.sentence_pause)

        if not finished:
            self.clarify_human_doubt(self._consume_latest_human())

        self.lines_since_last_human_interrupt += 1

    # ---------------- main run ----------------

    def run(self):
        story_lines, mid_q, mid_a, quiz = self.build_story()

        self.set_phase("GREETING")
        self.allow_miro_interrupt(False)
        self.speak_interruptible(
            "Hello everyone! How are you today? I am so excited! Today I will tell you a magical story!"
        )
        time.sleep(self.sentence_pause)

        self.set_phase("STORY_PART_1")
        self.allow_miro_interrupt(False)

        for idx, line in enumerate(story_lines[:12], start=1):
            self.deliver_story_line(line)

            if not self.miro_doubt_done:
                if idx >= 4 and self.should_trigger_miro_doubt_now():
                    self.miro_doubt_sequence()

        self.set_phase("MID_QUESTION")
        self.last_chunk_text = "%s (Correct: %s)" % (mid_q, mid_a)
        self.pub_last_chunk.publish(String(data=self.last_chunk_text))

        self.speak_interruptible("Okay now tell me this: " + mid_q)
        self.handle_question_with_human_priority(mid_q, mid_a, "MID::%s::%s" % (mid_q, mid_a))

        self.set_phase("STORY_PART_3")
        for line in story_lines[12:]:
            self.deliver_story_line(line)

        self.set_phase("QUIZ_INTRO")
        self.speak_interruptible("Okay now I will ask you some questions based on the story I just told.")
        self.set_phase("QUIZ")

        for i, item in enumerate(quiz[: self.quiz_count], start=1):
            q, correct = item
            self.last_chunk_text = "Question: %s | Correct: %s" % (q, correct)
            self.pub_last_chunk.publish(String(data=self.last_chunk_text))

            self.speak_interruptible("Question %d. %s" % (i, q))
            self.handle_question_with_human_priority(q, correct, "QUIZ::%d::%s::%s" % (i, q, correct))
            time.sleep(self.sentence_pause)

        self.set_phase("DONE")
        self.speak_interruptible("Great job everyone! The lecture is complete. Thank you and bye bye!")


if __name__ == "__main__":
    TeacherNode().run()
