// Create the cursor glow element
const cursorGlow = document.createElement('div');
cursorGlow.classList.add('cursor-glow');
document.body.appendChild(cursorGlow);

// Update the position of the cursor glow element
document.addEventListener('mousemove', (e) => {
  cursorGlow.style.opacity = 1;
  cursorGlow.style.top = `${e.pageY}px`;
  cursorGlow.style.left = `${e.pageX}px`;
});

// Hide the glow effect when the mouse stops moving
document.addEventListener('mouseleave', () => {
  cursorGlow.style.opacity = 0;
});

// Function to navigate to a different page
function navigateTo(page) {
  // Check if the page starts with '/'
  if (!page.startsWith('/')) {
    page = '/' + page;
  }
  window.location.href = page;
}

// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Attach click event listeners to elements with 'dash-container' class
  document.querySelectorAll('.dash-container').forEach(container => {
    container.addEventListener('click', function() {
      // Assuming the first child div of the container has the 'dash-text' class
      const page = this.getElementsByClassName('dash-text')[0].innerText.trim();
      switch (page) {
        case 'RF CODE 1':
          navigateTo('RF');
          break;
        case 'RF CODE 2':
          navigateTo('VAE');
          break;
        case 'RF CODE 3':
          navigateTo('GAN');
          break;
        case 'RF CODE 4':
          navigateTo('CNN');
          break;
        // Add more cases as needed
      }
    });
  });

  // Get all flash messages with the "flash-message" class
  var flashMessages = document.querySelectorAll(".flash-message");

  // Iterate through each flash message
  flashMessages.forEach(function (message) {
    // Set a timeout to hide the message after 1500 milliseconds (1.5 seconds)
    setTimeout(function () {
      message.style.display = "none";
    }, 1500);
  });
});

document.addEventListener('DOMContentLoaded', function () {
  function handleFormSubmitRF(event) {
      event.preventDefault();

      var formData = {
          api: document.getElementById('api').value,
          excipient: document.getElementById('excipient').value,
          api_percent: document.getElementById('api_percent').value,
          api_coated: document.getElementById('api_coated').checked,
          api_coat_percent: document.getElementById('api_coat_percent').value,
          exc_coated: document.getElementById('exc_coated').checked,
          exc_coat_percent: document.getElementById('exc_coat_percent').value,
          api_silica_type: document.getElementById('api_silica_type').value,
          exc_silica_type: document.getElementById('exc_silica_type').value
      };

      fetch('/predictRF', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
      })
      .then(response => response.json())
      .then(data => {
          document.getElementById('predictedFFRegime').textContent = '' + data.predicted_ff_regime;

          updateTimestamp();

          console.log('RF Form submitted and data updated.');
      })
      .catch((error) => {
        console.error('Error:', error);
        alert('An error occurred while submitting the RF form.');
    });
  }

  var rfForm = document.getElementById('dataFormRF');
  rfForm.addEventListener('submit', handleFormSubmitRF);
});

document.addEventListener('DOMContentLoaded', function () {
  let lastFormData = null;
  let lastResponseData = null;

  function handleFormSubmitVAE(event) {
      event.preventDefault();

      var formData = {
          api: document.getElementById('api').value,
          excipient: document.getElementById('excipient').value,
          api_percent: document.getElementById('api_percent').value,
          api_coat_percent: document.getElementById('api_coat_percent').value,
          exc_coat_percent: document.getElementById('exc_coat_percent').value,
          api_silica_type: document.getElementById('api_silica_type').value,
          exc_silica_type: document.getElementById('exc_silica_type').value
      };

      if (JSON.stringify(formData) === lastFormData) {
          console.log('Using cached response for identical input.');
          if (lastResponseData) {
              document.getElementById('predictedFFRegime').textContent = '' + lastResponseData.predicted_ff_regime;
              updateTimestamp();
              console.log('VAE Form submission used cached data.');
          }
          return;
      }

      console.log(formData);
      fetch('/predictVAE', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
      })
      .then(response => response.json())
      .then(data => {
          lastFormData = JSON.stringify(formData);
          lastResponseData = data;

          document.getElementById('predictedFFRegime').textContent = '' + data.predicted_ff_regime;

          updateTimestamp();

          console.log('VAE Form submitted and data updated.');
      })
      .catch((error) => {
          console.error('Error:', error);
          alert('An error occurred while submitting the VAE form.');
      });
  }

  var vaeForm = document.getElementById('dataFormVAE');
  vaeForm.addEventListener('submit', handleFormSubmitVAE);
});

function updateTimestamp() {
    let existingTimestamp = document.getElementById('predictionTimestamp');
    let currentTime = new Date().toLocaleTimeString();
    if (existingTimestamp) {
        existingTimestamp.textContent = 'Last updated: ' + currentTime;
    } else {
        let timestampElement = document.createElement('span');
        timestampElement.id = 'predictionTimestamp';
        timestampElement.className = 'prediction-timestamp';
        timestampElement.textContent = 'Last updated: ' + currentTime;
        
        let resultElement = document.getElementById('predictedFFRegime');
        resultElement.parentNode.insertBefore(timestampElement, resultElement.nextSibling);
    }
}
