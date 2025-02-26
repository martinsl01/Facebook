from PIL import Image
import os

def resize_image(final_size, im):
    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.Resampling.LANCZOS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

if __name__ == '__main__':

     if not os.path.exists('cleaned_images'):
        os.makedirs('cleaned_images')
        path = "C:/Users/marti/FMRRS/images_fb/images/"
        dirs = os.listdir(path)
        final_size = 512
        for n, item in enumerate(dirs, 0):
            im = Image.open("C:/Users/marti/FMRRS/images_fb/images/" + item)
            new_im = resize_image(final_size, im)
            new_im.save("C:/Users/marti/FMRRS/cleaned_images/" + item)
            