from enum import Enum


class Category(Enum):
    CLEAR_SKY = "A-Clear Sky"
    PATTERNED_CLOUDS = "B-Patterned Clouds"
    THIN_WHITE_CLOUDS = "C-Thin White Clouds"
    THICK_WHITE_CLOUDS = "D-Thick White Clouds"
    THICK_DARK_CLOUDS = "E-Thick Dark Clouds"
    VEIL_CLOUDS = "F-Veil Clouds"


label = {
    Category.CLEAR_SKY: 1,
    Category.PATTERNED_CLOUDS: 2,
    Category.THIN_WHITE_CLOUDS: 3,
    Category.THICK_WHITE_CLOUDS: 4,
    Category.THICK_DARK_CLOUDS: 5,
    Category.VEIL_CLOUDS: 6,
}
