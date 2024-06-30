import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path

def voice_to_voice(English):

    #transcribe audio
    transcription_response = audio_transcription(English)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        text = transcription_response.text

    te_translation, hi_translation, ta_translation, ml_translation, kn_translation = text_translation(text)

    te_audi_path = text_to_speech(te_translation) #Telugu
    hi_audi_path = text_to_speech(hi_translation) #Hindi
    ta_audi_path = text_to_speech(ta_translation) #Tamil
    ml_audi_path = text_to_speech(ml_translation) #Malayalam
    kn_audi_path = text_to_speech(kn_translation) #Kannada

    
    te_path = Path(te_audi_path)
    hi_path = Path(hi_audi_path)
    ta_path = Path(ta_audi_path)
    ml_path = Path(ml_audi_path)
    kn_path = Path(kn_audi_path)

    return te_path, hi_path, ta_path, ml_path, kn_path

def audio_transcription(English):

    aai.settings.api_key= "#"

    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(English)

    return transcription

def text_translation(text):

  translator_te = Translator(from_lang="en", to_lang="te") 
  te_text = translator_te.translate(text)

  translator_hi = Translator(from_lang="en", to_lang="hi") 
  hi_text = translator_hi.translate(text)

  translator_ta = Translator(from_lang="en", to_lang="ta") 
  ta_text = translator_ta.translate(text)

  translator_ml = Translator(from_lang="en", to_lang="ml") 
  ml_text = translator_ml.translate(text)

  translator_kn = Translator(from_lang="en", to_lang="kn") 
  kn_text = translator_kn.translate(text)

  return te_text, hi_text, ta_text, ml_text, kn_text


def text_to_speech(text): 

    client = ElevenLabs(
    api_key="#",
    )

    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=0.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    #File name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path



audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo= gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Telugu"), gr.Audio(label="Hindi"), gr.Audio(label="Tamil"), gr.Audio(label="Malayalam"), gr.Audio(label="Kannada")]
)

if __name__=="__main__":
    demo.launch()