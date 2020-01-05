import os
from flask import Flask, request, redirect, url_for, render_template,flash
from werkzeug.utils import secure_filename

from keras.models import Sequential,load_model
import keras,sys
from PIL import Image
import numpy as np
import warnings

warnings.simplefilter('ignore')
classes=["ニホンザル","イノシシ","カラス","ニワトリ","クマ","ゾウ","ウサギ"]
#classes=["monkey","boar","crow","chicken","bear","elephant","giraffe","lion","mouse","rabbit"]

num_classes=len(classes)
image_size=50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','gif'])

#appにインスタンス化
app=Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

#ファイル名が正しいか判定
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method=='POST':
        if 'file' not in request.files:
            #flash('ファイルがありません',"failed")
            return redirect(request.url)
            #return render_template('upload.html')
        file = request.files['file']
        if file.filename=='':
            #flash('ファイルがありません',"failed")
            return redirect(request.url)
            #return render_template('upload.html')
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            model=load_model('./animal_cnn_aug_add.h5')

            image=Image.open(filepath)
            image=image.convert('RGB')
            image=image.resize((image_size,image_size))
            data=np.asarray(image)
            X=[]
            X.append(data)
            X=np.array(X)

            result=model.predict([X])[0]
            predicted=result.argmax()
            percentage=int(result[predicted]*100)
            flag=predicted

            #return render_template('upload.html')
            return render_template('index.html',flag=flag)
            #return classes[predicted]+str(percentage)+"%"


            #return redirect(url_for('uploaded_file',filename=filename))
    return render_template('upload.html')

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__=='__main__':
    app.run()
