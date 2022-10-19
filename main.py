import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, flash
from scipy.interpolate import UnivariateSpline

app = Flask(__name__)
app.config['SECRET_KEY'] = "palavra-muito-secreta"

UPLOAD_FOLDER = os.path.abspath("static/img")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



###### PAGINAS ######
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sobre')
def sobre():
    return render_template('sobre.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/resultado')
def resultado():
    return render_template('resultado.html')



###### GERIAS ######
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload(): 
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = 'img.png'
        caminho = os.path.join(UPLOAD_FOLDER,filename)
        file.save(caminho)
        return redirect('/')
        
    elif file.filename == '':
        flash('Nenhuma imagem selecionada')
        return redirect('/')
        
    elif file != allowed_file(file.filename):
        flash('Formato de imagem inválido')
        return redirect('/')
        
    else :
        flash('Verifique o arquivo')
        return redirect('/')

def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))



###### FUNÇÕES ######
def grayscale(img):
    filtrada = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def chiado(img):
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = 0.8
    for i in range(height):
        for j in range(width):
            if np.random.rand() <= thresh:
                if np.random.randint(2) == 0:
                    gray[i, j] = min(gray[i, j] + np.random.randint(0, 64), 255)
                else: gray[i, j] = max(gray[i, j] - np.random.randint(0, 64), 0)
    filtrada = gray
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def lapis(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, (21,21), 0, 0)
    filtrada = cv2.divide(img_gray, 255 - img_blur, scale = 256)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def lapis2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_img = cv2.medianBlur(gray_img, 5)
    laplacian_img = cv2.Laplacian(median_img, cv2.CV_8U, ksize=5)
    _, thresh_img = cv2.threshold(laplacian_img, 100, 255, cv2.THRESH_BINARY_INV)
    pencilSketch_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    filtrada = pencilSketch_img
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def lapis3(img):
    pencilSketch_img = lapis2(img)
    pencilSketch_img = cv2.imread("static/img/filtrada.png")
    bilateral_img = cv2.bilateralFilter(img, 75, 100, 100)
    sketch_img = cv2.bitwise_and(bilateral_img, pencilSketch_img)
    filtrada = sketch_img
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def serpia(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza_normalizado = np.array(img_gray, np.float32)/255
    serpia = np.ones(img.shape)
    serpia[:,:,0] *= 153
    serpia[:,:,1] *= 204
    serpia[:,:,2] *= 255
    serpia[:,:,0] *= cinza_normalizado
    serpia[:,:,1] *= cinza_normalizado
    serpia[:,:,2] *= cinza_normalizado
    filtrada = np.array(serpia, np.uint8)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def contraste_preto(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-3, -5,  -3],
                    [-2, 0, 2],
                    [3, 5,  3]])
    filtrada = cv2.filter2D(img_gray, -1, kernel)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def saturado(img):
    ret, filtrada = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def inverter(img):
    filtrada = cv2.bitwise_not(img)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def detecta_face(img):
    xml = 'haarcascade_frontalface_alt2.xml'
    faceClassifier = cv2.CascadeClassifier(xml)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (205,90,106), 4)
    return cv2.imwrite("static/img/filtrada.png",img)

def censura_face(img):
    xml = 'haarcascade_frontalface_alt2.xml'
    faceClassifier = cv2.CascadeClassifier(xml)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(gray)
    for x,y,w,h in faces:
        roi = img[y:y+h, x:x+w] 
        roi = cv2.GaussianBlur(roi, (23, 23), 30) 
        img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    return cv2.imwrite("static/img/filtrada.png",img)

def quente(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    filtrada = cv2.merge((blue_channel, green_channel, red_channel ))
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def frio(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    filtrada = cv2.merge((blue_channel, green_channel, red_channel))
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def pb(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xp = [0, 64, 112, 128, 144, 192, 255]
    fp = [0, 16, 64, 128, 192, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    filtrada = cv2.LUT(gray, table)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def watercolor(img):
    filtrada = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return cv2.imwrite("static/img/filtrada.png",filtrada)

def emboss(img):
    height, width = img.shape[:2]
    y = np.ones((height, width), np.uint8) * 128
    output = np.zeros((height, width), np.uint8)
    kernel1 = np.array([[0, -1, -1],
                        [1, 0, -1],
                        [1, 1, 0]])
    kernel2 = np.array([[-1, -1, 0],
                        [-1, 0, 1],
                        [0, 1, 1]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output1 = cv2.add(cv2.filter2D(gray, -1, kernel1), y)
    output2 = cv2.add(cv2.filter2D(gray, -1, kernel2), y)
    for i in range(height):
        for j in range(width):
            output[i, j] = max(output1[i, j], output2[i, j])
    filtrada = output2
    return cv2.imwrite("static/img/filtrada.png",filtrada)



###### CHAMADA DAS FUNÇÕES ######
@app.route('/filtro_relevo', methods=['POST'])
def filtro_relevo():
    img_original = cv2.imread("static/img/img.png")
    emboss(img_original)
    return redirect('/resultado')

@app.route('/filtro_aquarela', methods=['POST'])
def filtro_aquarela():
    img_original = cv2.imread("static/img/img.png")
    watercolor(img_original)
    return redirect('/resultado')

@app.route('/filtro_pb', methods=['POST'])
def filtro_pb():
    img_original = cv2.imread("static/img/img.png")
    pb(img_original)
    return redirect('/resultado')

@app.route('/cinza', methods=['POST'])
def cinza():
    img_original = cv2.imread("static/img/img.png")
    grayscale(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_chiado', methods=['POST'])
def filtro_chiado():
    img_original = cv2.imread("static/img/img.png")
    chiado(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis', methods=['POST'])
def filtro_lapis():
    img_original = cv2.imread("static/img/img.png")
    lapis(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis2', methods=['POST'])
def filtro_lapis2():
    img_original = cv2.imread("static/img/img.png")
    lapis2(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis3', methods=['POST'])
def filtro_lapis3():
    img_original = cv2.imread("static/img/img.png")
    lapis3(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_serpia', methods=['POST'])
def filtro_serpia():
    img_original = cv2.imread("static/img/img.png")
    serpia(img_original)
    return redirect('/resultado')
    
@app.route('/contraste', methods=['POST'])
def contraste():
    img_original = cv2.imread("static/img/img.png")
    contraste_preto(img_original)
    return redirect('/resultado')
    
@app.route('/inverter_cores', methods=['POST'])
def inverter_cores():
    img_original = cv2.imread("static/img/img.png")
    inverter(img_original)
    return redirect('/resultado')
    
@app.route('/detectar_rosto', methods=['POST'])
def detectar_rosto():
    img_original = cv2.imread("static/img/img.png")
    detecta_face(img_original)
    return redirect('/resultado')
    
@app.route('/censura_rosto', methods=['POST'])
def censura_rosto():
    img_original = cv2.imread("static/img/img.png")
    censura_face(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_quente', methods=['POST'])
def filtro_quente():
    img_original = cv2.imread("static/img/img.png")
    quente(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_frio', methods=['POST'])
def filtro_frio():
    img_original = cv2.imread("static/img/img.png")
    frio(img_original)
    return redirect('/resultado')



###### RUN ######
if __name__ == '__main__':
    app.run(debug = True)