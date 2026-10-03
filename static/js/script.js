async function askQuestion() {
    const queryInput = document.getElementById("query");
    const responseDiv = document.getElementById("response");
    const query = queryInput.value.trim();

    if (!query) {
        responseDiv.innerHTML = "Please enter a question.";
        return;
    }

    responseDiv.innerHTML = "Thinking...";

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query
            })
        });

        const data = await res.json();

        let html = `
            <div class="answer">
                ${formatAnswer(data.response)}
            </div>
        `;

        if (data.sources && data.sources.length > 0) {
            html += `
                <div class="sources">
                    <h3>Sources</h3>
            `;

            data.sources.forEach((source, index) => {
                html += `
                    <div class="source">
                        <div class="source-title">
                            [${index + 1}] ${source.book_name || "Unknown Book"}
                        </div>

                        <div>
                            <strong>Chapter:</strong>
                            ${source.chapter_title || "Unknown Chapter"}
                        </div>

                        <div>
                            <strong>Page:</strong>
                            ${source.page}
                        </div>
                        <div class="source-text">
                            ${source.text}
                        </div>
                    </div>
                `;
            });

            html += `
                </div>
            `;
        }
        responseDiv.innerHTML = html;
    } catch (error) {
        console.error(error);
        responseDiv.innerHTML =
            "Something went wrong while generating the answer.";
    }
}

function formatAnswer(text) {
    let formatted = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    formatted = formatted
        .replace(/\\\*/g, "*")
        .replace(/\\_/g, "_")
        .replace(/\\(?=\n|$)/g, "");

    formatted = formatted.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );

    formatted = formatted.replace(
        /\*([^*\n]+)\*/g,
        "$1"
    );

    formatted = formatted.replace(
        /\*/g,
        ""
    );

    formatted = formatted.replace(
        /\n/g,
        "<br>"
    );

    return formatted;
}
