const joke_element = document.getElementById("joke_element");
            const get_joke_button = document.getElementById("get_joke_button");

            function fetchJoke() {
                fetch("https://icanhazdadjoke.com/slack")
                    .then(data => data.json())
                    .then(jokeData => {
                        const joketext = jokeData.attachments[0].text;
                        joke_element.innerHTML = joketext;
                    });
            }

            get_joke_button.addEventListener("click", fetchJoke);

            // Initial joke fetch on page load
            fetchJoke();