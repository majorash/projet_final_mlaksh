$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        var form_data = new FormData($('#upload-file')[0]);

        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        // $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

    $('#btn-search').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        // $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/reverse_search',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                var obj = JSON.parse(data);
                console.log(obj)
                // var values = Object.values(data)
                // console.log(values);
                // jQuery.each(json, function(i, val) {
                //     // console.log(i);
                //     console.log(val);
                //   });
                console.log(obj.length)
                for(var i = 0; i< obj.length;i++){
                    var img = $("<img />").attr('src', obj[i]);
                    console.log(img)
                    $(img).appendTo('#image-grid');
                }
                // $('#result').text(' Result:  ' + data);
                console.log('Success!');
            },
        });
    });

});
