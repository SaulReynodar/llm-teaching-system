#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
import sounddevice as sd
import numpy as np
import whisper
import tempfile
import wave
import os
import time


class STTNode:
    def __init__(self):
        rospy.init_node("stt_node")

        self.pub = rospy.Publisher("/human/text", String, queue_size=10)

        self.model_size = rospy.get_param("~model_size", "base")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.record_seconds = float(rospy.get_param("~record_seconds", 2.0))
        self.energy_threshold = float(rospy.get_param("~energy_threshold", 0.03))
        self.device = rospy.get_param("~device", None)
        self.language = rospy.get_param("~language", "en")

        self.miro_speaking = False
        self.teacher_started = False

        rospy.Subscriber("/miro/speaking", Bool, self.miro_speaking_callback)
        rospy.Subscriber("/teacher/state", String, self.teacher_state_callback)

        rospy.loginfo("Loading Whisper model: %s", self.model_size)
        self.model = whisper.load_model(self.model_size)

        rospy.loginfo("STT ready. It will listen during class, including while teacher is speaking.")

    def miro_speaking_callback(self, msg):
        self.miro_speaking = bool(msg.data)

    def teacher_state_callback(self, msg):
        state = (msg.data or "").strip()
        if state in ["SPEAKING", "IDLE"]:
            self.teacher_started = True

    def should_listen(self):
        return self.teacher_started and not self.miro_speaking

    def record_audio(self):
        frames = int(self.sample_rate * self.record_seconds)

        audio = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device
        )

        sd.wait()
        audio = np.squeeze(audio)

        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
        return audio, rms

    def save_wav(self, audio):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)

            audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(audio_i16.tobytes())

        return path

    def run(self):
        while not rospy.is_shutdown():
            try:
                if not self.should_listen():
                    time.sleep(0.05)
                    continue

                rospy.loginfo("[STT] Listening...")
                audio, rms = self.record_audio()

                if not self.should_listen():
                    continue

                if rms < self.energy_threshold:
                    rospy.loginfo("[STT] Too quiet (rms=%.4f). Skipping.", rms)
                    continue

                wav_path = self.save_wav(audio)

                rospy.loginfo("[STT] Transcribing...")
                result = self.model.transcribe(wav_path, language=self.language)

                if os.path.exists(wav_path):
                    os.remove(wav_path)

                text = (result.get("text") or "").strip()

                if text and self.should_listen():
                    rospy.loginfo("[STT] You said: %s", text)
                    self.pub.publish(String(data=text))
                else:
                    rospy.loginfo("[STT] No usable human text detected.")

                time.sleep(0.1)

            except Exception as e:
                rospy.logerr("STT error: %s", str(e))
                time.sleep(0.3)


if __name__ == "__main__":
    node = STTNode()
    node.run()
