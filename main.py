import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, flash
from scipy.interpolate import UnivariateSpline
import pytesseract

app = Flask(__name__)
app.config['SECRET_KEY'] = "palavra-muito-secreta"

CAMINHO_ABS = os.path.join('/home/thauan/Desktop/APS/Site/static/img')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

pasta_img = "/home/thauan/Desktop/APS/Site/static/img"


###### PAGINAS ######
#RETORNA O END-POINT DAS PAGINAS
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

@app.route('/texto')
def texto():
    return render_template('texto.html')

@app.route('/extracao-de-dados')
def dados():
    return render_template('extracao-de-dados.html')



###### GERIAS ######
#VERIFICA A EXTENSÃO DO ARQUIVO
def allowed_file(filename): #FUNÇÃO DE EXTENSÕES PERMITIDAS
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #VERIFICA SE APÓS O PONTO A EXTENSÃO DO ARQUIVO BATE COM OS VALORES DECLARADOS COMO PERMITIDOS

@app.route('/upload', methods=['POST'])
def upload(): 
    file = request.files['file'] #RECEBE O ARQUIVO COMO SENDO TIPO ARQUVO
    if file and allowed_file(file.filename): #SE RECEBER UM ARQUIVO E A EXTENSÃO DO ARQUIVO FOR PERMITIDA, CAI NESSE IF
        filename = secure_filename(file.filename) #SELECIONA O NOME DO ARQUIVO
        filename = 'img.png' #ALTERA O NOME DO ARQUIVO
        caminho = os.path.join(CAMINHO_ABS,filename) #UTILIZANDO O CAMINHO ABSOLUTO DECLARADO, CRIA UM NOVO CAMINHO COM O NOME DO ARQUIVO (EXEMPLO: C:/ARQUIVOS/IMAGEM.PNG)
        file.save(caminho) #SALVA O ARQUIVO NO CAMINHO CRIADO
        return redirect('/') #REDIRECIONA O USUARIO PARA PAGINA INDEX ONDE ELE VISUALIZARÁ SUA IMAGEM SALVA
        
    elif file.filename == '': #SE NENHUMA IMAGEM FOR SELECIONADA
        flash('Nenhuma imagem selecionada') #CRIA UMA MENSAGEM FLASH INFORMANDO AO USUARIO QUE NÃO TEM FOI SELECIONADA NENHUMA IMAGEM
        return redirect('/') #REDIRECIONA PARA PAGINA INDEX COM A MENSAGEM FLASH EXIBIDA
        
    elif file != allowed_file(file.filename): #SE A EXTENSÃO DO ARQUIVO NÃO FOR PERMITIDA
        flash('Formato de imagem inválido') #CRIA UMA MENSAGEM FLASH INFORMANDO AO USUARIO QUE O TIPO DE ARQUIVO SELECIONADO NÃO É DO FORMATO PERMITIDO
        return redirect('/') #REDIRECIONA PARA PAGINA INDEX COM A MENSAGEM FLASH EXIBIDA
        
    else : #SE PASSAR POR TODOS OS IFs (INCLUINDO O QUE SALVA A IMAGEM DO USUARIO)
        flash('Verifique o arquivo') #CRIA UMA MENSAGEM FLASH GENERALIZADA, PARA QUE O USUARIO VERIFIQUE O ARQUIVO, JÁ QUE NÃO DEU ERRO DE SELEÇÃO, NEM DE EXTENSÃO, MAS TAMBEM NÃO SALVOU A IMAGEM ORIGINAL
        return redirect('/') #REDIRECIONA PARA PAGINA INDEX COM A MENSAGEM FLASH EXIBIDA

def LookupTable(x, y): 
  spline = UnivariateSpline(x, y) #"spline é uma curva definida matematicamente por dois ou mais pontos de controle"
  return spline(range(256)) #RETORNA A CURVA NO RANGE DE 1byte



###### FUNÇÕES ######
#FAZ O BOTÃO FUNCIONAR
def grayscale(img):
    filtrada = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #TRANSFORMA A IMAGEM ORIGINAL EM CINZA
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada) #SALVA A IMAGEM

def chiado(img):
    h, w = img.shape[:2] #ALTERA O FORMATO DA IMAGEM EM ALTURA E LARGURA
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CRIA UMA NOVA IMAGEM EM CINZA
    thresh = 0.8 #VARIAVEL DECLARANDO VALOR
    for i in range(h): #PARA CADA PIXEL EM ALTURA
        for j in range(w): #PARA CADA PIXEL EM LARGURA
            if np.random.rand() <= thresh: #SE O VALOR RANDOM FOR MENOR QUE THRESH (0.8)
                if np.random.randint(2) == 0: #SE O VALOR INTEIRO FOR MENOR QUE 0
                    img_gray[i, j] = min(img_gray[i, j] + np.random.randint(0, 64), 255) #TROCA A COR DO PIXEL PARA CLARO
                else: img_gray[i, j] = max(img_gray[i, j] - np.random.randint(0, 64), 0) #TROCA A COR DO PIXEL PARA ESCURO
    filtrada = img_gray #VARIAVEL QUE RECEBE A IMAGEM FILTRADA
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada) #SALVA A IMAGEM FILTRADA

def lapis(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CONVERTE A IMAGEM EM ESCALA DE CINZA
    img_gray_inv = 255 - img_gray #INVERTE A COR DA IMAGEM
    img_blur = cv2.GaussianBlur(img_gray_inv, (21,21), 0, 0)
    filtrada = cv2.divide(img_gray, 255 - img_blur, scale = 256)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def lapis2(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_img = cv2.medianBlur(gray_img, 5)
    laplacian_img = cv2.Laplacian(median_img, cv2.CV_8U, ksize=5)
    _, thresh_img = cv2.threshold(laplacian_img, 100, 255, cv2.THRESH_BINARY_INV)
    pencilSketch_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    filtrada = pencilSketch_img
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def lapis3(img):
    pencilSketch_img = lapis2(img)
    pencilSketch_img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    bilateral_img = cv2.bilateralFilter(img, 75, 100, 100)
    sketch_img = cv2.bitwise_and(bilateral_img, pencilSketch_img)
    filtrada = sketch_img
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def serpia(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza_normalizado = np.array(img_gray, np.float32)/255
    serpia = np.ones(img.shape)
    serpia[:,:,0] *= 153
    serpia[:,:,1] *= 210
    serpia[:,:,2] *= 255
    serpia[:,:,0] *= cinza_normalizado
    serpia[:,:,1] *= cinza_normalizado
    serpia[:,:,2] *= cinza_normalizado
    filtrada = np.array(serpia, np.uint8)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def contraste_preto(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-3, -5,  -3],
                    [-2, 0, 2],
                    [3, 5,  3]])
    filtrada = cv2.filter2D(img_gray, -1, kernel)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def saturado(img):
    ret, filtrada = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def inverter(img):
    filtrada = cv2.bitwise_not(img)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def quente(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    filtrada = cv2.merge((blue_channel, green_channel, red_channel ))
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def frio(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    filtrada = cv2.merge((blue_channel, green_channel, red_channel))
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def pb(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xp = [0, 64, 112, 128, 144, 192, 255]
    fp = [0, 16, 64, 128, 192, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    filtrada = cv2.LUT(gray, table)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

def watercolor(img):
    filtrada = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)

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
    filtrada = output
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/filtrada.png",filtrada)


### DADOS ### 
def detecta_face(img):
    xml = '/home/thauan/Desktop/APS/Site/haarcascades/haarcascade_frontalface_alt2.xml' #MAPEANDO ARQUIVO XML
    encontrar_rosto = cv2.CascadeClassifier(xml) #CARREGA O ARQUIVO XML PARA UTILIZAR COMO FUNÇÃO DO OPENCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CONVERTE A IMAGEM PARA ESCALA CINZA
    rosto = encontrar_rosto.detectMultiScale(gray_img) #DETECTA A FACE USANDO A IMAGEM CINZA
    for x,y,w,h in rosto:#VARIAVEIS PARA CONSTRUIR O RETANGULO NA FACE DETECTADA
        #X = EIXO X / Y = EIXO Y / W = WIDTH (LARGURA) / H = HEIGHT (ALTURA)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2) #CONSTROI UM RETANGULO NA IMAGEM ORIGINAL UTILIZANDO AS MEDIDAS DA VARIAVEL ENCONTRADAS NA IMAGEM CINZA
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/dados.png",img) #SALVA A IMAGEM 

def censura_face(img):
    xml = '/home/thauan/Desktop/APS/Site/haarcascades/haarcascade_frontalface_alt2.xml' #MAPEANDO ARQUIVO XML
    encontrar_rosto = cv2.CascadeClassifier(xml) #CARREGA O ARQUIVO XML PARA UTILIZAR COMO FUNÇÃO DO OPENCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CONVERTE A IMAGEM PARA ESCALA CINZA
    rosto = encontrar_rosto.detectMultiScale(gray_img) #DETECTA A FACE USANDO A IMAGEM CINZA
    for x,y,w,h in rosto:
        #X = EIXO X / Y = EIXO Y / W = WIDTH (LARGURA) / H = HEIGHT (ALTURA)
        roi = img[y:y+h, x:x+w] #USANDO AS VARIAVEIS, RECORTA DA IMAGEM ORIGINAL A FACE ENCONTRADA NA IMAGEM CINZA
        roi = cv2.GaussianBlur(roi, (23, 23), 30) #APLICA BORRÃO NA IMAGEM RECORTADA
        img[y:y+roi.shape[0], x:x+roi.shape[1]] = roi #ADICIONA O RECORTE NA IMAGEM ORIGINAL
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/dados.png",img) #SALVA A IMAGEM 

def recorta_face(img):
    xml = '/home/thauan/Desktop/APS/Site/haarcascades/haarcascade_frontalface_alt2.xml' #MAPEANDO ARQUIVO XML
    encontrar_rosto = cv2.CascadeClassifier(xml) #CARREGA O ARQUIVO XML PARA UTILIZAR COMO FUNÇÃO DO OPENCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CONVERTE A IMAGEM PARA ESCALA CINZA
    rosto = encontrar_rosto.detectMultiScale(gray_img) #DETECTA A FACE USANDO A IMAGEM CINZA
    for x,y,w,h in rosto: #PARA CADA MEDIDA NO ROSTO ENCONTRADO:
        #X = EIXO X / Y = EIXO Y / W = WIDTH (LARGURA) / H = HEIGHT (ALTURA)
        face = img[y:y+h, x:x+w] #RECORTA O ROSTO DA IMAGEM ORIGINAL USANDO AS VARIAVEIS
        img = face #ROSTO RECORTADO É DECLARADO COMO IMAGEM ORIGINAL
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/dados.png",img) #SALVA A IMAGEM 

def detecta_gato(img):
    xml = '/home/thauan/Desktop/APS/Site/haarcascades/haarcascade_frontalface_alt2.xml' #MAPEANDO ARQUIVO XML (DETECTAR GATO)
    encontrar_rosto = cv2.CascadeClassifier(xml) #CARREGA O ARQUIVO XML PARA UTILIZAR COMO FUNÇÃO DO OPENCV
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #CONVERTE A IMAGEM PARA ESCALA CINZA
    rosto_gato = encontrar_rosto.detectMultiScale(gray_img) #DETECTA A FACE USANDO A IMAGEM CINZA
    for (x, y,w,h) in rosto_gato: #PARA CADA MEDIDA NO ROSTO ENCONTRADO:
        #X = EIXO X / Y = EIXO Y / W = WIDTH (LARGURA) / H = HEIGHT (ALTURA)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2) #CONSTROI O RETANGULO NA FACE DETECTADA
    return cv2.imwrite("/home/thauan/Desktop/APS/Site/static/img/dados.png",img) #SALVA A IMAGEM 

def transcrever_txt(img):
    #pytesseract.pytesseract.tesseract_cmd = r"C:\\Tesseract-OCR\tesseract.exe" #VARIAVEL PRA INDICAR O CONSOLE DO PYTESSERACT
    texto_encontrado = pytesseract.pytesseract.image_to_string(img,lang='por') #TRANSFORMA A IMAGEM EM TEXTO
    if len(texto_encontrado) <= 3:
        print("nao encontrou nada")
        texto_encontrado = 'Não foi encontrado nenhum texto!'
    return flash(texto_encontrado) #RETORNA O TEXTO PARA PAGINA DO SITE



###### CHAMADA DAS FUNÇÕES ######
@app.route('/filtro_relevo', methods=['POST']) #CODIGO FLASK PARA UTILIZAR FUNÇÃO
def filtro_relevo(): #FUNÇÃO DO PYTHON
    img_original = cv2.imread(CAMINHO_ABS,'img.png') #LÊ A IMAGEM DO USUARIO
    emboss(img_original) #CHAMA A FUNÇÃO PASSANDO A IMAGEM ORIGINAL COMO PARAMETRO
    return redirect('/resultado') #RETORNA PARA PAGINA RESULTADO COM A IMAGEM MANIPULADA

@app.route('/filtro_aquarela', methods=['POST'])
def filtro_aquarela():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    watercolor(img_original)
    return redirect('/resultado')

@app.route('/filtro_pb', methods=['POST'])
def filtro_pb():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    pb(img_original)
    return redirect('/resultado')

@app.route('/cinza', methods=['POST'])
def cinza():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    grayscale(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_chiado', methods=['POST'])
def filtro_chiado():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    chiado(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis', methods=['POST'])
def filtro_lapis():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    lapis(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis2', methods=['POST'])
def filtro_lapis2():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    lapis2(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_lapis3', methods=['POST'])
def filtro_lapis3():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    lapis3(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_serpia', methods=['POST'])
def filtro_serpia():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    serpia(img_original)
    return redirect('/resultado')
    
@app.route('/contraste', methods=['POST'])
def contraste():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    contraste_preto(img_original)
    return redirect('/resultado')
    
@app.route('/inverter_cores', methods=['POST'])
def inverter_cores():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    inverter(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_quente', methods=['POST'])
def filtro_quente():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    quente(img_original)
    return redirect('/resultado')
    
@app.route('/filtro_frio', methods=['POST'])
def filtro_frio():
    img_original = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/img.png")
    frio(img_original)
    return redirect('/resultado')
    

### EXTRAÇÃO DE DADOS ###
@app.route('/detectar_rosto', methods=['POST'])
def detectar_rosto():
    img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    detecta_face(img)
    return redirect('/extracao-de-dados')
    
@app.route('/censura_rosto', methods=['POST'])
def censura_rosto():
    img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    censura_face(img)
    return redirect('/extracao-de-dados')
    
@app.route('/corta_rosto', methods=['POST'])
def corta_rosto():
    img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    recorta_face(img)
    return redirect('/extracao-de-dados')
    
@app.route('/rosto_gato', methods=['POST'])
def rosto_gato():
    img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    detecta_gato(img)
    return redirect('/extracao-de-dados')

@app.route('/escreve_img', methods=['POST'])
def escreve_img():
    img = cv2.imread("/home/thauan/Desktop/APS/Site/static/img/filtrada.png")
    transcrever_txt(img)
    return redirect('/texto')


###### RUN ######
if __name__ == '__main__':
    app.run(debug = True)