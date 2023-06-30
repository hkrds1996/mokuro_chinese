# mokuro translator

Read Japanese manga with selectable text inside a browser.

This project is a personal branch from  [https://github.com/kha-white/mokuro](https://github.com/kha-white/mokuro "mokuro"). Mokuro has already given good function. However, for learning Japanese by using Manga, if a translator is combined in html, it will be much better. Then, I added a translator function to the Mokuro project.

# Display
https://github.com/hkrds1996/mokuro_chinese/assets/16051414/9b618241-23ca-49ac-b812-196bea5b58b1


# Install
Requirements are the same as the original Mokuro project.

Instead of using pip3 install, we should do the following command   
```commandline
python setup.py install
```

# Using
Using way is the same as the Mokuro project
If want to change translated language, change dest's value

```python
def translate_to_chinese(text):
    translator = Translator()
    translation = translator.translate(text, dest='zh-cn')
    return translation.text
```
# Contact
For any inquiries, please feel free to contact me at fasklas68@gmail.com

# BV
https://www.bilibili.com/video/BV1vm4y1a7o3

# Acknowledgments
- https://github.com/kha-white/mokuro
- https://github.com/dmMaze/comic-text-detector
- https://github.com/juvian/Manga-Text-Segmentation
