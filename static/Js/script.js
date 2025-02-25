// Get elements
const getStarted = document.getElementById('getStarted');
const banner = document.getElementById('banner');
const panel = document.getElementById('panel');

// Event Listener for "Get Started" button
getStarted.addEventListener('click', () => {
  banner.classList.add('hidden'); // Hide the banner
  panel.classList.add('active'); // Show the panel
});


// Speech synthesis setup
const synth = window.speechSynthesis;
let voices = [];
let selectedVoice = null;

function loadVoices() {
  voices = synth.getVoices();
  console.log('Available Voices:', voices); // Log all voices available in the browser
  
  // Male voice
  const maleVoice = voices.find(voice => voice.name.toLowerCase().includes('google uk english male') || voice.name.toLowerCase().includes('google english male')) || voices[0];  // Default male voice
  
  // Female voice
  const femaleVoice = voices.find(voice => voice.name.toLowerCase().includes('google uk english female') || voice.name.toLowerCase().includes('google english female')) || voices[0];  // Default female voice

  console.log('Male Voice:', maleVoice.name);
  console.log('Female Voice:', femaleVoice.name);

  // Default to male voice for Avatar 1 and female voice for Avatar 2
  selectedVoice = maleVoice;
}

// Load voices when they are available
if (speechSynthesis.onvoiceschanged !== undefined) {
  speechSynthesis.onvoiceschanged = loadVoices;
} else {
  loadVoices();
}

// Avatar selection
const avatarVideo = document.getElementById('avatarVideo');
const prevAvatar = document.getElementById('prevAvatar');
const nextAvatar = document.getElementById('nextAvatar');
const avatarVideos = [
  "static/avatars/Avatar1.mp4", // Avatar 1 video (Male voice)
  "static/avatars/Avatar2.mp4", // Avatar 2 video (Female voice)
];

let avatarIndex = 0;

function updateAvatarAndVoice() {
  // Update the video source
  avatarVideo.src = avatarVideos[avatarIndex];
  avatarVideo.load(); // Reload the video
  
  // Set the selected voice based on the avatar
  if (avatarIndex === 0) {
    // Male voice for Avatar 1
    selectedVoice = voices.find(voice => voice.name.toLowerCase().includes("google uk english male") || voice.name.toLowerCase().includes("google english male")) || voices[0];
  } else {
    // Female voice for Avatar 2
    selectedVoice = voices.find(voice => voice.name.toLowerCase().includes("google uk english female") || voice.name.toLowerCase().includes("google english female")) || voices[2];
  }
  
  console.log(`Selected Voice for Avatar ${avatarIndex + 1}: ${selectedVoice.name}`);
}

// Handle Previous Avatar Button
prevAvatar.addEventListener('click', () => {
  avatarIndex = (avatarIndex - 1 + avatarVideos.length) % avatarVideos.length;
  updateAvatarAndVoice();
});

// Handle Next Avatar Button
nextAvatar.addEventListener('click', () => {
  avatarIndex = (avatarIndex + 1) % avatarVideos.length;
  updateAvatarAndVoice();
});

// Text-to-speech functionality
const speakBtn = document.getElementById('speakBtn');
speakBtn.addEventListener('click', () => {
  const text = document.getElementById('predictedText').textContent;
  if (!text) {
    alert('No text available to speak!');
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = selectedVoice; // Set the voice explicitly

  // Play the video when speaking starts
  avatarVideo.play();

  // Pause the video when speech ends
  utterance.onend = () => {
    avatarVideo.pause();
    avatarVideo.currentTime = 0;
  };

  // Speak the text
  synth.speak(utterance);
});

// Submit functionality
// Submit functionality
document.getElementById('uploadForm').addEventListener('submit', async function(event) {
  event.preventDefault();

  const formData = new FormData();
  const imageFile = document.getElementById('image').files[0];
  const txtFile = document.getElementById('txt').files[0];

  formData.append('image', imageFile);
  if (txtFile) {
      formData.append('txt', txtFile);
  }
  try {
    const response = await fetch("/submit", {
      method: "POST",
      body: formData,
    });

    const results = await response.text();

    const outputDiv = document.getElementById('predictedText');
    outputDiv.style.display = 'block';
    if (response.ok) {
        outputDiv.innerHTML = `<pre>${results}</pre>`;
    } else {
        outputDiv.textContent = `Error: ${results.error}`;
    }
  } catch (error) {
      console.error('Error:', error);
  }
});
