// IMPORTAR O ARQUIVO
function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#file_upload') 
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// BOTÕES DE FUNÇÕES
var btn1 = document.getElementById('btn1');
var btn2 = document.getElementById('btn2');

function switch1() {
    if (dados.classList.toggle('hide')) {
        filtros.classList.toggle('hide')
        btn1.classList.toggle('show-btn')
        btn2.classList.toggle('show-btn')
        document.getElementById("center").style.height = "364px"
    } else if (filtros.classList.toggle('hide')){
        dados.classList.toggle('hide')
        btn1.classList.toggle('show-btn')
        document.getElementById("center").style.height = "0px"
    } else {
        dados.classList.toggle('hide')
        btn1.classList.toggle('show-btn')
        document.getElementById("center").style.height = "364px"
    }
}

function switch2() {
    if (filtros.classList.toggle('hide')) {
        dados.classList.toggle('hide')
        btn2.classList.toggle('show-btn')
        btn1.classList.toggle('show-btn')
        document.getElementById("center").style.height = "248px"
    } else if (dados.classList.toggle('hide')){
        filtros.classList.toggle('hide')
        btn2.classList.toggle('show-btn')
        document.getElementById("center").style.height = "0px"
    } else {
        filtros.classList.toggle('hide')
        btn2.classList.toggle('show-btn')
        document.getElementById("center").style.height = "248px"
    }
}

// BORDA - SOBRE E TUTORIAL
function aumentaBorder() {
    document.getElementById("op1").style.borderLeft = "5px solid #013243";
    document.getElementById("op1").style.marginLeft = "0px";

    document.getElementById("op2").style.borderLeft = "1px solid #013243";
    document.getElementById("op2").style.marginLeft = "2px";

    document.getElementById("op3").style.borderLeft = "1px solid #013243";
    document.getElementById("op3").style.marginLeft = "2px";

    document.getElementById("op4").style.borderLeft = "1px solid #013243";
    document.getElementById("op4").style.marginLeft = "2px";

    document.getElementById("op5").style.borderLeft = "1px solid #013243";
    document.getElementById("op5").style.marginLeft = "2px";

    document.getElementById("op6").style.borderLeft = "1px solid #013243";
    document.getElementById("op6").style.marginLeft = "2px";

    }

function aumentaBorder2() {
    document.getElementById("op1").style.borderLeft = "1px solid #013243";
    document.getElementById("op1").style.marginLeft = "2px";

    document.getElementById("op2").style.borderLeft = "5px solid #013243";
    document.getElementById("op2").style.marginLeft = "0px";

    document.getElementById("op3").style.borderLeft = "1px solid #013243";
    document.getElementById("op3").style.marginLeft = "2px";

    document.getElementById("op4").style.borderLeft = "1px solid #013243";
    document.getElementById("op4").style.marginLeft = "2px";

    document.getElementById("op5").style.borderLeft = "1px solid #013243";
    document.getElementById("op5").style.marginLeft = "2px";

    document.getElementById("op6").style.borderLeft = "1px solid #013243";
    document.getElementById("op6").style.marginLeft = "2px";
    }

function aumentaBorder3() {
    document.getElementById("op1").style.borderLeft = "1px solid #013243";
    document.getElementById("op1").style.marginLeft = "2px";

    document.getElementById("op2").style.borderLeft = "1px solid #013243";
    document.getElementById("op2").style.marginLeft = "2px";

    document.getElementById("op3").style.borderLeft = "5px solid #013243";
    document.getElementById("op3").style.marginLeft = "0px";

    document.getElementById("op4").style.borderLeft = "1px solid #013243";
    document.getElementById("op4").style.marginLeft = "2px";

    document.getElementById("op5").style.borderLeft = "1px solid #013243";
    document.getElementById("op5").style.marginLeft = "2px";
    
    document.getElementById("op6").style.borderLeft = "1px solid #013243";
    document.getElementById("op6").style.marginLeft = "2px";
    }


function aumentaBorder4() {
    document.getElementById("op1").style.borderLeft = "1px solid #013243";
    document.getElementById("op1").style.marginLeft = "2px";

    document.getElementById("op2").style.borderLeft = "1px solid #013243";
    document.getElementById("op2").style.marginLeft = "2px";

    document.getElementById("op3").style.borderLeft = "1px solid #013243";
    document.getElementById("op3").style.marginLeft = "2px";

    document.getElementById("op4").style.borderLeft = "5px solid #013243";
    document.getElementById("op4").style.marginLeft = "0px";

    document.getElementById("op5").style.borderLeft = "1px solid #013243";
    document.getElementById("op5").style.marginLeft = "2px";
    
    document.getElementById("op6").style.borderLeft = "1px solid #013243";
    document.getElementById("op6").style.marginLeft = "2px";
    }


function aumentaBorder5() {
    document.getElementById("op1").style.borderLeft = "1px solid #013243";
    document.getElementById("op1").style.marginLeft = "2px";

    document.getElementById("op2").style.borderLeft = "1px solid #013243";
    document.getElementById("op2").style.marginLeft = "2px";

    document.getElementById("op3").style.borderLeft = "1px solid #013243";
    document.getElementById("op3").style.marginLeft = "2px";

    document.getElementById("op4").style.borderLeft = "1px solid #013243";
    document.getElementById("op4").style.marginLeft = "2px";

    document.getElementById("op5").style.borderLeft = "5px solid #013243";
    document.getElementById("op5").style.marginLeft = "0px";
    
    document.getElementById("op6").style.borderLeft = "1px solid #013243";
    document.getElementById("op6").style.marginLeft = "2px";
    }
    
function aumentaBorder6() {
    document.getElementById("op1").style.borderLeft = "1px solid #013243";
    document.getElementById("op1").style.marginLeft = "2px";

    document.getElementById("op2").style.borderLeft = "1px solid #013243";
    document.getElementById("op2").style.marginLeft = "2px";

    document.getElementById("op3").style.borderLeft = "1px solid #013243";
    document.getElementById("op3").style.marginLeft = "2px";

    document.getElementById("op4").style.borderLeft = "1px solid #013243";
    document.getElementById("op4").style.marginLeft = "2px";

    document.getElementById("op5").style.borderLeft = "1px solid #013243";
    document.getElementById("op5").style.marginLeft = "2px";

    document.getElementById("op6").style.borderLeft = "5px solid #013243";
    document.getElementById("op6").style.marginLeft = "0px";
    }