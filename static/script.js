// // Get elements
// const getStarted = document.getElementById('getStarted');
// const banner = document.getElementById('banner');
// const panel = document.getElementById('panel');

// // Event Listener for "Get Started" button
// getStarted.addEventListener('click', () => {
//   banner.classList.add('hidden'); // Hide the banner
//   panel.classList.add('active'); // Show the panel
// });


// // Speech synthesis setup
// const synth = window.speechSynthesis;
// let voices = [];
// let selectedVoice = null;

// function loadVoices() {
//   voices = synth.getVoices();
//   console.log('Available Voices:', voices); // Log all voices available in the browser
  
//   // Male voice
//   const maleVoice = voices.find(voice => voice.name.toLowerCase().includes('google uk english male') || voice.name.toLowerCase().includes('google english male')) || voices[0];  // Default male voice
  
//   // Female voice
//   const femaleVoice = voices.find(voice => voice.name.toLowerCase().includes('google uk english female') || voice.name.toLowerCase().includes('google english female')) || voices[0];  // Default female voice

//   console.log('Male Voice:', maleVoice.name);
//   console.log('Female Voice:', femaleVoice.name);

//   // Default to male voice for Avatar 1 and female voice for Avatar 2
//   selectedVoice = maleVoice;
// }

// // Load voices when they are available
// if (speechSynthesis.onvoiceschanged !== undefined) {
//   speechSynthesis.onvoiceschanged = loadVoices;
// } else {
//   loadVoices();
// }

// // Avatar selection
// const avatarVideo = document.getElementById('avatarVideo');
// const prevAvatar = document.getElementById('prevAvatar');
// const nextAvatar = document.getElementById('nextAvatar');
// const avatarVideos = [
//   "/avatars/Avatar1.mp4", // Avatar 1 video (Male voice)
//   "/avatars/Avatar2.mp4", // Avatar 2 video (Female voice)
// ];

// let avatarIndex = 0;

// function updateAvatarAndVoice() {
//   // Update the video source
//   avatarVideo.src = avatarVideos[avatarIndex];
//   avatarVideo.load(); // Reload the video
  
//   // Set the selected voice based on the avatar
//   if (avatarIndex === 0) {
//     // Male voice for Avatar 1
//     selectedVoice = voices.find(voice => voice.name.toLowerCase().includes("google uk english male") || voice.name.toLowerCase().includes("google english male")) || voices[0];
//   } else {
//     // Female voice for Avatar 2
//     selectedVoice = voices.find(voice => voice.name.toLowerCase().includes("google uk english female") || voice.name.toLowerCase().includes("google english female")) || voices[2];
//   }
  
//   console.log(`Selected Voice for Avatar ${avatarIndex + 1}: ${selectedVoice.name}`);
// }

// // Handle Previous Avatar Button
// prevAvatar.addEventListener('click', () => {
//   avatarIndex = (avatarIndex - 1 + avatarVideos.length) % avatarVideos.length;
//   updateAvatarAndVoice();
// });

// // Handle Next Avatar Button
// nextAvatar.addEventListener('click', () => {
//   avatarIndex = (avatarIndex + 1) % avatarVideos.length;
//   updateAvatarAndVoice();
// });

// // Text-to-speech functionality
// const speakBtn = document.getElementById('speakBtn');
// speakBtn.addEventListener('click', () => {
//   const text = document.getElementById('textContent').textContent;
//   if (!text) {
//     alert('No text available to speak!');
//     return;
//   }

//   const utterance = new SpeechSynthesisUtterance(text);
//   utterance.voice = selectedVoice; // Set the voice explicitly

//   // Play the video when speaking starts
//   avatarVideo.play();

//   // Pause the video when speech ends
//   utterance.onend = () => {
//     avatarVideo.pause();
//     avatarVideo.currentTime = 0;
//   };

//   // Speak the text
//   synth.speak(utterance);
// });

// // Submit functionality
// document.getElementById("submitBtn").addEventListener("click", async () => {
//   const imageInput = document.getElementById("imageUpload");
//   if (!imageInput.files.length) {
//     alert("Please upload an image first.");
//     return;
//   }

//   const formData = new FormData();
//   formData.append("image", imageInput.files[0]);

//   try {
//     const response = await fetch("/predict", {
//       method: "POST",
//       body: formData,
//     });

//     const data = await response.json();
//     if (data.predicted_text) {
//       document.getElementById("textContent").textContent = data.predicted_text;
//     } else {
//       alert("Prediction failed: " + data.error);
//     }
//   } catch (error) {
//     console.error("Error:", error);
//   }
// });


// Get elements
const getStarted = document.getElementById('getStarted');
const banner = document.getElementById('banner');
const panel = document.getElementById('panel');
const avatarVideo = document.getElementById('avatarVideo');
const prevAvatar = document.getElementById('prevAvatar');
const nextAvatar = document.getElementById('nextAvatar');
const speakBtn = document.getElementById('speakBtn');
const playButton = document.getElementById('playButton');
const pauseButton = document.getElementById('pauseButton');
const muteButton = document.getElementById('muteButton'); // Speaker icon
const submitButton = document.getElementById("submitBtn");

const avatarVideos = [
  "/avatars/Avatar1.mp4", // Avatar 1 (Male voice)
  "/avatars/Avatar2.mp4"  // Avatar 2 (Female voice)
];

let avatarIndex = 0;
let synth = window.speechSynthesis;
let voices = [];
let selectedVoice = null;
let isMuted = false;
let currentSpeech = null;

// Event Listener for "Get Started" button
if (getStarted) {
  getStarted.addEventListener('click', () => {
    banner.classList.add('hidden'); // Hide the banner
    panel.classList.add('active'); // Show the panel
  });
}

// Load voices
function loadVoices() {
  voices = synth.getVoices();
  
  if (voices.length === 0) {
    console.warn("No voices available. Retry voice loading.");
    setTimeout(loadVoices, 500); // Retry loading voices
    return;
  }

  console.log('Available Voices:', voices);

  selectedVoice = voices.find(voice => voice.name.includes('Male')) || voices[0];

  updateAvatarAndVoice(); // Ensure voice updates when avatar changes
}

// Ensure voices load properly
if (speechSynthesis.onvoiceschanged !== undefined) {
  speechSynthesis.onvoiceschanged = loadVoices;
} else {
  loadVoices();
}

// Function to update avatar & voice
function updateAvatarAndVoice() {
  avatarVideo.src = avatarVideos[avatarIndex];
  avatarVideo.load();

  selectedVoice = avatarIndex === 0 
    ? voices.find(voice => voice.name.includes("Male")) || voices[0] 
    : voices.find(voice => voice.name.includes("Female")) || voices[1];

  console.log(`Selected Voice for Avatar ${avatarIndex + 1}: ${selectedVoice ? selectedVoice.name : "Default"}`);
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

// Function to start speech synthesis
function startSpeech(text) {
  if (!text || isMuted) return;

  if (synth.speaking) synth.cancel(); // Stop any existing speech

  currentSpeech = new SpeechSynthesisUtterance(text);
  currentSpeech.voice = selectedVoice;
  currentSpeech.lang = 'en-US';
  currentSpeech.rate = 1;

  avatarVideo.play(); // Start avatar animation when speaking

  currentSpeech.onend = () => {
    avatarVideo.pause();
    avatarVideo.currentTime = 0;
  };

  synth.speak(currentSpeech);
}

// Play Button - Starts or Resumes Speech
playButton.addEventListener('click', () => {
  const text = document.getElementById('textContent').textContent;
  if (text.trim()) {
    if (synth.paused) {
      synth.resume();
    } else {
      startSpeech(text);
    }
  }
});

// Pause Button - Pauses Speech
pauseButton.addEventListener('click', () => {
  if (synth.speaking) {
    synth.pause();
  }
});

// Mute Button - Toggles Speech
muteButton.addEventListener('click', () => {
  isMuted = !isMuted;
  muteButton.innerHTML = isMuted ? "🔇" : "🔊"; // Change icon
  if (synth.speaking) synth.cancel(); // Stop speaking when muted
});

// Text-to-Speech on Click
speakBtn.addEventListener('click', () => {
  const text = document.getElementById('textContent').textContent;
  startSpeech(text);
});

// Handle Form Submission for Image Prediction
submitButton.addEventListener("click", async () => {
  const imageInput = document.getElementById("imageUpload");
  if (!imageInput.files.length) {
    alert("Please upload an image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (data.predicted_text) {
      document.getElementById("textContent").textContent = data.predicted_text;
      startSpeech(data.predicted_text); // Speak after prediction
    } else {
      alert("Prediction failed: " + data.error);
    }
  } catch (error) {
    console.error("Error:", error);
  }
});
