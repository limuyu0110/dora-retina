from flask import Flask,render_template, url_for, send_file, request, redirect
import requests
import base64

app = Flask(__name__)


@app.route('/doraeye')
def hello_world():
    return render_template('eye.html', score='0', filename='')


@app.route('/update/',  methods=['post', 'get'])
def upload():
    print request.method
    if request.method == 'GET':
        return render_template('eye.html', score='0', filename='')
    else:
        f = request.files.get("files")
        if not f:
            return render_template('eye.html', score='0', filename='')
        im = f.read()
        fn = f.filename
        print fn
        f.seek(0)
        f.save('img/%s' % fn)
        f.close()
        im = base64.b64encode(im)
        print 'receive img'
        try:
            r = requests.post('http://23.96.49.78', data={'data':im})
            score = r.content 
            print r.content
        except Exception as e:
            score = 'error'
            print e
        return render_template('eye.html', score=score, filename=fn)


@app.route('/static2/',  methods=['get'])
def static2():
    fn = request.args.get("filename")
    fn = 'img/%s' % fn
    print 'filename----------', fn
    return send_file(fn, mimetype='image/png')



def main():
     app.run()
if __name__ == '__main__':
    main()
