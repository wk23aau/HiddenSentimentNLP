document.getElementById("startBtn").addEventListener("click", function() {
    chrome.runtime.sendMessage({ action: "start" }, response => {
      console.log("Start clicked:", response);
    });
  });
  
  document.getElementById("pauseBtn").addEventListener("click", function() {
    chrome.runtime.sendMessage({ action: "pause" }, response => {
      console.log("Pause clicked:", response);
    });
  });
  
  document.getElementById("stopBtn").addEventListener("click", function() {
    chrome.runtime.sendMessage({ action: "stop" }, response => {
      console.log("Stop clicked:", response);
    });
  });
  