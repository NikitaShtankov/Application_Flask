import flask
from flask import render_template
import pickle
import sklearn

app = flask.Flask(__name__, template_folder='source')

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('lr_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
            with open('scaler_y.pkl', 'rb') as s:
                scaler = pickle.load(s)

        input1 = float(flask.request.form['Прочность при растяжении'])
        input2 = float(flask.request.form['Плотность, кг/м3'])
        input3 = float(flask.request.form['Модуль упругости, ГПа'])
        input4 = float(flask.request.form['Количество отвердителя, м.%'])
        input5 = float(flask.request.form['Содержание эпоксидных групп,%_2'])
        input6 = float(flask.request.form['Температура вспышки, С_2'])
        input7 = float(flask.request.form['Поверхностная плотность, г/м2'])
        input8 = float(flask.request.form['Модуль упругости при растяжении, ГПа'])
        input9 = float(flask.request.form['Потребление смолы, г/м2'])
        input10 = float(flask.request.form['Угол нашивки, град'])
        input11 = float(flask.request.form['Шаг нашивки'])
        input12 = float(flask.request.form['Плотность нашивки'])

        y_pred = loaded_model.predict([[input1,input2,input3,input4,input5,input6,input7,input8,input9,input10,input11,input12]])
        pred = scaler.inverse_transform(y_pred)
        pred = pred[0,0]
        return render_template('main.html', result=pred)

if __name__ == '__main__':
    app.run()
