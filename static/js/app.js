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

function showHide(id) {
    let center = document.querySelector(id);
    center.classList.toggle("show");
}

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
    }