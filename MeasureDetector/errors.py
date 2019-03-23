class InvalidImageFormatError(Exception):
    """Raised, when the image format is neither JPEG nor PNG"""
    pass


class InvalidImageError(Exception):
    """Raised, when the image is invalid or could not be loaded"""
    pass


class InvalidAnnotationError(Exception):
    """Raised, when the annotations are invalid"""
    pass