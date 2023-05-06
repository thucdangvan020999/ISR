import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('models/RealESRGAN_x2.pth', download=True)
model4 = RealESRGAN(device, scale=4)
model4.load_weights('models/RealESRGAN_x4.pth', download=True)
model8 = RealESRGAN(device, scale=8)
model8.load_weights('models/RealESRGAN_x8.pth', download=True)


def inference(image, size):
    if size == '2x':
        result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        result = model4.predict(image.convert('RGB'))
    else:
        result = model8.predict(image.convert('RGB'))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


title = "Face Real ESRGAN UpScale: 2x 4x 8x"
description = "This is an unofficial demo for Real-ESRGAN. Scales the resolution of a photo. This model shows better results on faces compared to the original version.<br>Telegram BOT: https://t.me/restoration_photo_bot"
article = "<div style='text-align: center;'>Twitter <a href='https://twitter.com/DoEvent' target='_blank'>Max Skobeev</a> | <a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a>  <center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_face_esrgan' alt='visitor badge'></center></div>"

title = "Face Real ESRGAN UpScale: 2x 4x 8x"
description = "This is an unofficial demo for Real-ESRGAN. Scales the resolution of a photo. This model shows better results on faces compared to the original version.<br>Telegram BOT: https://t.me/restoration_photo_bot"
article = "<div style='text-align: center;'>Twitter <a href='https://twitter.com/DoEvent' target='_blank'>Max Skobeev</a> | <a href='https://huggingface.co/sberbank-ai/Real-ESRGAN' target='_blank'>Model card</a>  <center><img src='https://visitor-badge.glitch.me/badge?page_id=max_skobeev_face_esrgan' alt='visitor badge'></center></div>"


gr.Interface(inference,
    [gr.Image(type="pil"), 
    gr.Radio(['2x', '4x', '8x'], 
    type="value",
    value='2x',
    label='Resolution model')], 
    gr.Image(type="pil", label="Output"),
    article=article,
    examples=[['./uploads/54.png', "2x"]],
    allow_flagging='never',
    cache_examples=False,
    ).queue(concurrency_count=1, api_open=False).launch(show_api=False, show_error=True)
    