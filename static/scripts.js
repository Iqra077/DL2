document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("recommendationForm");
  const loader = document.getElementById("loader");
  const resultSection = document.getElementById("resultSection");
  const recommendationContainer = document.getElementById("recommendationContainer");

  form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const studentId = document.getElementById("studentId").value;

      // Show loader
      loader.classList.remove("hidden");
      resultSection.classList.add("hidden");
      recommendationContainer.innerHTML = "";

      try {
          const response = await fetch(`/recommendations?student_id=${studentId}`);
          const data = await response.json();

          if (data.error) {
              alert(data.error);
          } else {
              resultSection.classList.remove("hidden");

              // Display performance level and average score
              const { recommendations, performance_level, average_score } = data;
              const info = document.createElement("div");
              info.className = "info";

              info.innerHTML = `
                  <h3>Performance Level: ${performance_level}</h3>
                  <p><strong>Average Score:</strong> ${average_score}</p>
              `;
              recommendationContainer.appendChild(info);

              // Populate recommendations
              recommendations.forEach((rec) => {
                  const card = document.createElement("div");
                  card.className = "card";

                  card.innerHTML = `
                      <h3>${rec.Type}</h3>
                      <p><strong>Platform:</strong> ${rec.Platform}</p>
                      <a href="${rec.Resource_Link}" target="_blank">Access Resource</a>
                  `;
                  recommendationContainer.appendChild(card);
              });
          }
      } catch (error) {
          alert("Failed to fetch recommendations. Please try again.");
      } finally {
          // Hide loader
          loader.classList.add("hidden");
      }
  });
});
