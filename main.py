from pyannote.audio import Pipeline
import io

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")


def test():
    with open("audio.wav", 'rb') as fd:
        contents = fd.read()
        diarization = pipeline(io.BytesIO(contents))


if __name__ == '__main__':
    test()

