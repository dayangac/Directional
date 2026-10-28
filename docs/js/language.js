//C++ and Python buttons in the header, which choose the language of every code tab on the site, as in the Polyscope
//documentation. The choice is remembered across pages.
(function () {
    var LANGUAGES = ["C++", "Python"];

    function readLanguage() {
        try {
            return localStorage.getItem("directional-language") || "C++";
        } catch (e) {
            return "C++";
        }
    }

    function setLanguage(language) {
        try {
            localStorage.setItem("directional-language", language);
        } catch (e) {}
        document.querySelectorAll(".tabbed-set").forEach(function (tabs) {
            tabs.querySelectorAll(".tabbed-labels > label").forEach(function (label) {
                if (label.textContent.trim()===language)
                    document.getElementById(label.htmlFor).checked = true;
            });
        });
        document.querySelectorAll(".lang-button").forEach(function (button) {
            button.classList.toggle("lang-button-inactive", button.textContent!==language);
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        var source = document.querySelector(".md-header__source");
        LANGUAGES.forEach(function (language) {
            var button = document.createElement("button");
            button.className = "lang-button";
            button.textContent = language;
            button.addEventListener("click", function () { setLanguage(language); });
            source.parentNode.insertBefore(button, source);
        });
        setLanguage(readLanguage());
    });
})();
