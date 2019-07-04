import music_analysis
import os


ma = music_analysis.MusicAnalysis(
    "/home/m_umerjavaid/Compare-music/training/checkpoints/attributes/resnet50_13_syn_0.80_cat_0.90.pt")
base_path = "/home/m_umerjavaid/Compare-music/training/datasets/TestM/series3/"
extensions = ["series3_00002_C.wav", "series3_00002_F.wav",
              "series3_00002_P.wav", "series3_00002_R.wav", "series3_00002_V.wav"]
paths = list()
for extension in extensions:
    paths.append(os.path.join(base_path, extension))
ma.process_music_chunks(paths, 1)
