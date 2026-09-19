from PIL import Image

img = Image.open("data/images/B.png").convert("RGBA")

background = Image.new(
    "RGBA",
    img.size,
    (255, 255, 255, 255)
)

background.alpha_composite(img)

background.convert("RGB").save(
    "data/images/B_fixed.png"
)

print("saved: data/images/B_fixed.png")