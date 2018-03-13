var express = require('express');
var multer = require('multer');
var fs = require('fs');
var bodyParser = require('body-parser');
var pythonShell = require('python-shell');
var async = require('async');

var _storage = multer.diskStorage({
  destination: function(req, file, cb){
    //cb(null, 'python/')
    cb(null, 'data/demo/')
  },
  filename: function(req, file, cb){
    cb(null, Date.now() + '_' + file.originalname)
  }
})
var upload = multer({storage: _storage}).any();

var app = express();

var options = {
  mode: 'text',
  pythonPath: '',
  pythonOptions: ['-u'],
  scriptPath: 'ctpn/',
  args: ['']
};

// create application/json parser
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json({ type: 'application/*+json' }));

app.locals.pretty = true;
app.set('view engine', 'jade');
app.set('views', './views');

app.use('/user', express.static('uploads'));
app.use(express.static('public'));
app.use(express.static('python'));
app.get('/', function(req, res){
    res.render('index')
});

app.post('/translate', function(req, res){
  var tasks = [
    function(callback){
      upload(req,res,function(err, result) {
        //options.args[0] = JSON.stringify(req.body.source_lang); 
        python_args = [req.files[0].filename, req.body.source_lang, req.body.target_lang];
         callback(null, python_args);
      });

    },
    function(data, callback){
      var options = {
          mode: 'text',
          pythonPath: '',
          pythonOptions: ['-u'],
          scriptPath: 'ctpn/', 
          args: data
      };

      pythonShell.run('demo.py', options, function (err, results) {
        //if (err) throw err;
        console.log(options);
        console.log('results: %j', results);
        res.render('result',{results: results})
      });
      callback(null);
    }
  ];
  async.waterfall(tasks, function(err){});
});
app.get('/temp', function(req, res){
    //res.render('temp')
    pythonShell.run('font.py', options, function (err, results) {
      if (err) throw err;
      console.log('results: %j', results);
    });
});

app.post('/guide', function (req, res) {
    console.log(req.body)
    res.send(req.body)
});

app.get('/result', function (req, res) {
    res.render('result')
});




app.get('/InTextgram', function(req, res){
    res.send('Hello InText, <h1>original</h1><img src="/Image/12.jpg">');
    //res.send('Hello InText, <h1>tesseract result</h1>');
    var child = require('child_process').execFile;
    var executablePath = "C:\\Users\\sm-pc\\Documents\\InTextgram\\exetest\\end_to_end.exe";//"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe";
    var parameters = ["C:\\Users\\sm-pc\\Documents\\InTextgram\\public\\Image\\12.jpg"];//["--incognito"];

    child(executablePath, parameters, function(err, data) {
         console.log(err)
         console.log(data.toString());
    });
});
app.get('/ResultInText', function(req, res){

    res.send('Hello InText, <h1>original</h1><img src="/Image/original.jpg"> <h1>recognition result</h1><img src="/Image/recognition.jpg">');
    //res.send('<h1>Login please</h1>');
});

app.get('/upload', function(req, res){
    res.render('upload');
});

app.post('/upload', function(req,res){
    var child = require('child_process').execFile;
    var executablePath = "C:\\Users\\sm-pc\\Documents\\InTextgram\\exetest\\end_to_end.exe";//"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe";
    var parameters = ["C:\\Users\\sm-pc\\Documents\\InTextgram\\public\\Image\\"+req.file.filename];//["--incognito"];

    console.log(parameters);

    child(executablePath, parameters, function(err, data) {
         console.log(err)
         console.log(data.toString());
    });

    res.send('<a href="/ResultInText">see result</a>');
    //res.send('Hello InText, <h1>original</h1><img src="/Image/'+req.file.filename+'"> <h1>recognition result</h1><img src="/Image/recognition.jpg">');
    //res.send('Uploaded : '+req.file.filename);
});

app.listen(3000, function(){
    console.log('Conneted 3000 port!');
});
