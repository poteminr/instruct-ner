from __future__ import unicode_literals
import youtube_dl
import os
import whisper


def download_video_from_youtube(title: str, url: str) -> None:
    ydl_opts = {'outtmpl': '{}.%(ext)s'.format(title), 'nocheckcertificate':True}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_audio(video_path: str, audiofile_name: str) -> None:
    os.system('ffmpeg -y  -threads 4\
 -i {} -f wav -ab 192000 -vn {}'.format(video_path, audiofile_name))


def get_transcript(audio_path: str) -> str:
    model = whisper.load_model('small')
    result = model.transcribe(audio_path)
    return result['text']

