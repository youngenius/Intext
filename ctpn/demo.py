#!/usr/bin/env python
#-*- coding: utf-8 -*-#
#from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import glob
import shutil
import pytesseract
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
#translate
from googleapiclient.discovery import build
import json
#draw
from PIL import ImageFont, ImageDraw, Image
#dpi
from PIL import Image

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

#trim function add
def im_trim(img, x, y, w, h, n):
    x -= 5
    y -= 5
    h += 10
    w += 10
    img_trim = img[y:y+h, x:x+w]
    file_name = "%d.jpg" % (n)
    cv2.imwrite(file_name, img_trim)
    return img_trim

#translate function add
def translate(text, target_lang):
      service = build('translate', 'v2',
            developerKey='AIzaSyAACjxpSuCRKpzALRrNkC49RN4ZAtuJsUA')

      response = service.translations().list(
      #source='en',
	      target=target_lang,
	      model='nmt',
	      q= text
      ).execute()
      
      #for result in response['translations']:
      #	print('detect: ' + result['detectedSourceLanguage']+ ', '
      #      +'translatedText: ' + result['translatedText'])


      return response['translations'][0]['translatedText']

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[0] - i, coordinates[1] - i)
        rect_end = (coordinates[6] + i, coordinates[7] + i)
        draw.rectangle((rect_start, rect_end), outline = color)

#draw text
def draw_text(text, img_cv, x1, y1, x2, y2):
    #print(text)
    #img_cv = cv2.imread('12.jpg')
    #img = Image.open("12.jpg")
    img_pil = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_pil)

    box = [x1, y1, x2, y2]

    # use a truetype font
    
    font = ImageFont.truetype("NanumSquareB.ttf", 60)
    text_size = font.getsize(text)

    textX = (box[2] - text_size[0]) / 2
    textY = box[1] - text_size[1]

    #draw rectangle
    draw = ImageDraw.Draw(img)
    draw_rectangle(draw, box, color='red', width=5)
    draw.text((textX, textY), text , font=font, fill='red')
    
    return img

def draw_boxes(img,image_name,boxes,scale,source_lang, target_lang):
    base_name = image_name.split('/')[-1]
    i = 0
    all_text = ''
    img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cv = Image.fromarray(img_pil)
    with open('data/results/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)

            #cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            #cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            #cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            #cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
		
	    #print(int(box[0])) 	    
	    #image crop
	    trim_image = im_trim(img, (int(box[0])), (int(box[1])), (int(box[2])-int(box[0])), (int(box[5])-int(box[1])), i)	    

	    #dpi change for tesseract
 	    image_name = "%d.jpg" % (i)
	    trim_image = Image.open(image_name)
	    trim_image.save("dpichange.png", dpi=(300,300))		
	    trim_image = Image.open('dpichange.png')
	
	    #tesseract
	    #print(trim_image.info)	    
	    text = pytesseract.image_to_string(trim_image, source_lang)
	    #print('tesseract: '+text)
	    
	    #draw trans_text
	    trans_text = translate(text, target_lang)
	    all_text += text
	   
	    #img = draw_text(trans_text, img, int(box[0]), int(box[1]), int(box[6]), int(box[7]))

	    # use a truetype font
	    
	    font = ImageFont.truetype("NanumSquareB.ttf", trim_image.height/3)
	    text_size = font.getsize(trans_text)

	    textX = int(box[0])#(int(box[6]) - text_size[0]) / 2
	    textY = int(box[1]) - text_size[1]

	    #draw rectangle
	    draw = ImageDraw.Draw(img_cv)
	    draw_rectangle(draw, box, color='red', width=2)
	    draw.text((textX, textY), trans_text , font=font, fill='red')	    
	
	    i+= 1
            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line)

    result_image_path = os.path.join("python",'test.jpg')
    img_cv.save(os.path.join("python","test.jpg"))
    #img_cv.show()
    #print("All translate-----------")
    all_text = translate(all_text, target_lang)	

    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data/results", base_name), img)

def ctpn(sess, net, image_name, source_lang, target_lang):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    all_text = draw_boxes(img, image_name, boxes, scale, source_lang, target_lang)
    timer.toc()
    #print(('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0]))    


if __name__ == '__main__':    
    #main()

    # print command line arguments
    #for arg in sys.argv[1:]:
    #    print(arg)
    
    file_name = sys.argv[1] 
    source_lang = sys.argv[2]
    target_lang = sys.argv[3]
    
    #print(file_name)
    #print(source_lang)
    #print(target_lang)
    
    if source_lang=="lang_detect":
	source_lang = "eng+chi_sim+chi_tra+kor+jpn"
	
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    #print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        #print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        #print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    #im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #           glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', file_name)) 

    for im_name in im_names:
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name, source_lang, target_lang)
