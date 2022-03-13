from torchlm.data import LandmarksWFLWConverter


def test_torchlm_data_converter():
    wflw_converter = LandmarksWFLWConverter(
        wflw_dir="../data/WFLW",
        save_dir="../data/WFLW/convertd",
        extend=0.2,
        rebuild=True,
        target_size=256,
        keep_aspect=False,
        force_normalize=True,
        force_absolute_path=True
    )
    wflw_converter.convert()

    wflw_converter.show(count=30, original=False)


if __name__ == "__main__":
    test_torchlm_data_converter()
    """
    python3 ./data.py
    """
