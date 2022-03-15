from torchlm.data import LandmarksWFLWConverter

def test_torchlm_data_converter():
    converter = LandmarksWFLWConverter(
        data_dir="../data/WFLW",
        save_dir="../data/WFLW/converted",
        extend=0.2,
        rebuild=True,
        target_size=256,
        keep_aspect=False,
        force_normalize=True,
        force_absolute_path=True
    )
    converter.convert()

    converter.show(count=30)


if __name__ == "__main__":
    test_torchlm_data_converter()
    """
    python3 ./data.py
    """
