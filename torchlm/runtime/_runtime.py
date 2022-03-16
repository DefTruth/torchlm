# helpers from other modules (usage: from ..runtime import)
# _runtime: use this script to avoid circular imports in 'runtime' module
from ..core import FaceDetBase, LandmarksDetBase
from ..models.pipnet import get_meanface, DEFAULT_MEANFACE_STRINGS