from torchlm.data import LandmarksWFLWConverter
from torchlm.data import Landmarks300WConverter
from torchlm.data import LandmarksCOFWConverter
from torchlm.data import LandmarksAFLWConverter

def test_wflw_converter():
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


def test_300w_converter():
    converter = Landmarks300WConverter(
        data_dir="../data/300W",
        save_dir="../data/300W/converted",
        extend=0.2,
        rebuild=True,
        target_size=256,
        keep_aspect=False,
        force_normalize=True,
        force_absolute_path=True
    )
    converter.convert()

    converter.show(count=30)


def test_cofw_converter():
    converter = LandmarksCOFWConverter(
        data_dir="../data/COFW",
        save_dir="../data/COFW/converted",
        extend=0.2,
        rebuild=True,
        target_size=256,
        keep_aspect=False,
        force_normalize=True,
        force_absolute_path=True
    )
    converter.convert()

    converter.show(count=30)

def test_aflw_converter():
    converter = LandmarksAFLWConverter(
        data_dir="../data/AFLW",
        save_dir="../data/AFLW/converted",
        extend=0.2,
        rebuild=True,
        target_size=256,
        keep_aspect=False,
        force_normalize=True,
        force_absolute_path=True
    )
    converter.convert()

    converter.show(count=10)

if __name__ == "__main__":
    test_wflw_converter()
    test_300w_converter()
    test_cofw_converter()
    test_aflw_converter()
    """
    python3 ./data.py
    """
