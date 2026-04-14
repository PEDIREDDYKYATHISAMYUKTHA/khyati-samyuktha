# This file contains hairstyle recommendation logic

def get_hairstyles(face_shape):
    """
    Returns hairstyle suggestions based on detected face shape.
    """

    styles = {
        "Oval": [
            "Layered Cut",
            "Textured Quiff",
            "Pompadour",
            "Undercut",
            "Side Part"
        ],

        "Round": [
            "High Fade",
            "Spiky Hair",
            "Faux Hawk",
            "Angular Fringe",
            "Side Swept"
        ],

        "Square": [
            "Classic Taper",
            "Crew Cut",
            "Side Part",
            "Short Layers",
            "Textured Crop"
        ],

        "Oblong": [
            "Fringe Cut",
            "Medium Layers",
            "Side Part",
            "Low Fade",
            "Messy Style"
        ]
    }

    return styles.get(face_shape, ["No suggestions available"])
