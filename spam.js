$(document).ready(function(){
    $("#submitBtn").click(function(){
        var inputValue = $("#inputValue").val();
        console.log(inputValue);
        
        // Make a POST request with inputValue as payload
        fetch('http://localhost:5000/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // body: JSON.stringify({ inputValue: inputValue })
            body: JSON.stringify({ message: messageValue })

        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // Display the result in the HTML
            $("#result").text("Server response: " + data.result);
        })
        .catch(error => {
            console.error('Error:', error);
            $("#result").text("An error occurred: " + error.message);
        });
    });
});