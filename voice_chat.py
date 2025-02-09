import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS
import soundfile as sf
import os
from pydub import AudioSegment
from pydub.playback import play

import datetime

asr_cache_dir="cache/whisper"
os.makedirs(asr_cache_dir,exist_ok=True)
os.environ["WHISPER_CACHE_DIR"]=asr_cache_dir

llm_cache_dir = "cache/huggingface"
os.makedirs(llm_cache_dir, exist_ok=True)  # もしディレクトリがなければ作成
os.environ["TRANSFORMERS_CACHE"] = llm_cache_dir

tts_cache_dir="cache/tts"
os.makedirs(tts_cache_dir,exist_ok=True)
os.environ["TTS_CACHE_DIR"]=tts_cache_dir



# ------------------------------
# 1. 音声認識 (ASR) - Whisper
# ------------------------------
whisper_model = whisper.load_model("small",download_root=os.environ["WHISPER_CACHE_DIR"])  # small, medium, large などを選択可能

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# ------------------------------
# 2. 自然言語理解＆生成 (NLU & NLG) - LLaMA
# ------------------------------
llama_model_name = "elyza/ELYZA-japanese-Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name,cache_dir=os.environ["TRANSFORMERS_CACHE"])
model = AutoModelForCausalLM.from_pretrained(llama_model_name,cache_dir=os.environ["TRANSFORMERS_CACHE"]).to("cuda" if torch.cuda.is_available() else "cpu")

def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ------------------------------
# 3. 音声合成 (TTS) - Tacotron2
# ------------------------------
tts = TTS("tts_models/ja/kokoro/tacotron2-DDC").to("cuda" if torch.cuda.is_available() else "cpu")

def synthesize_speech(text, output_dir="responses"):
    """ AIの返答をWAVファイルとして保存 """
    # 保存先フォルダを作成
    os.makedirs(output_dir, exist_ok=True)

    # タイムスタンプ付きのファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"response_{timestamp}.wav")

    # 音声合成
    tts.tts_to_file(text=text, file_path=output_file)
    
    return output_file  # 保存したWAVファイルのパスを返す

# ------------------------------
# 4. 音声対話の実行
# ------------------------------
def voice_chat(audio_path):
    print("[1] 音声認識中...")
    user_text = transcribe_audio(audio_path)
    print(f"ユーザー: {user_text}")

    print("[2] 返答生成中...")
    response_text = generate_response(user_text)
    print(f"AI: {response_text}")

    print("[3] 音声合成中...")
    response_audio = synthesize_speech(response_text)

    print("[4] 音声を再生...")
    sound = AudioSegment.from_wav(response_audio)
    play(sound)

    return response_audio

# ------------------------------
# 実行
# ------------------------------
if __name__ == "__main__":
    input_audio = "sample_input.wav"  # 入力音声ファイル（事前に録音して用意）
    voice_chat(input_audio)
