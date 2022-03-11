from torchlm.data import LandmarksWFLWConverter


def test_torchlm_data_converter():
    wflw_converter = LandmarksWFLWConverter(
        wflw_dir="../data/WFLW",
        save_dir="../data/WFLW/convertd",
        extend=0.2,
        rebuild=False,
        force_normalize=False,
        force_absolute_path=False
    )
    wflw_converter.convert()


if __name__ == "__main__":
    test_torchlm_data_converter()

    """
    python3 ./data.py
    """
