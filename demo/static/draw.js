var context = document.getElementById('canvas').getContext("2d");
context.globalCompositeOperation = 'luminosity';
var canvas = document.getElementById('canvas');

// Get coordinates for both mouse and touch events
function getEventPos(e) {
  var rect = canvas.getBoundingClientRect();
  var scaleX = canvas.width / rect.width;
  var scaleY = canvas.height / rect.height;
  
  var clientX, clientY;
  
  if (e.type.indexOf('touch') !== -1) {
    clientX = e.touches[0] ? e.touches[0].clientX : e.changedTouches[0].clientX;
    clientY = e.touches[0] ? e.touches[0].clientY : e.changedTouches[0].clientY;
  } else {
    clientX = e.clientX;
    clientY = e.clientY;
  }
  
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

// Mouse events
$('#canvas').mousedown(function(e){
  e.preventDefault();
  var pos = getEventPos(e);
  paint = true;
  addClick(pos.x, pos.y);
  redraw();
});

$('#canvas').mousemove(function(e){
  e.preventDefault();
  if(paint){
    var pos = getEventPos(e);
    addClick(pos.x, pos.y, true);
    redraw();
  }
});

$('#canvas').mouseup(function(e){
  e.preventDefault();
  paint = false;
});

$('#canvas').mouseleave(function(e){
  paint = false;
});

// Touch events for mobile
$('#canvas').on('touchstart', function(e){
  e.preventDefault();
  var pos = getEventPos(e.originalEvent);
  paint = true;
  addClick(pos.x, pos.y);
  redraw();
});

$('#canvas').on('touchmove', function(e){
  e.preventDefault();
  if(paint){
    var pos = getEventPos(e.originalEvent);
    addClick(pos.x, pos.y, true);
    redraw();
  }
});

$('#canvas').on('touchend touchcancel', function(e){
  e.preventDefault();
  paint = false;
});

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;

function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

function redraw(){
  //context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas

  context.strokeStyle = "#fff";
  context.fillStyle = "#000";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  context.lineJoin = "round";
  context.lineWidth = 10;

  for(var i=0; i < clickX.length; i++) {
    context.beginPath();
    if(clickDrag[i] && i){
      context.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       context.moveTo(clickX[i]-1, clickY[i]);
     }
     context.lineTo(clickX[i], clickY[i]);
     context.closePath();
     context.stroke();
  }
}
