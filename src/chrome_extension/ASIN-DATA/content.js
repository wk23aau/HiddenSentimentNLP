(function() {
    // Environment variables (load from .env in your build process)
    const GITHUB_OWNER = process.env.REACT_APP_GITHUB_OWNER;
    const GITHUB_REPO = process.env.REACT_APP_GITHUB_REPO;
    const GITHUB_BRANCH = process.env.REACT_APP_GITHUB_BRANCH || "main";
    const GITHUB_TOKEN = process.env.REACT_APP_GITHUB_TOKEN;
  
    // Unicode-safe Base64 encoding (from Reddit script)
    function base64EncodeUnicode(str) {
      return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, 
        function(match, p1) {
          return String.fromCharCode(parseInt(p1, 16));
        }
      ));
    }
  
    // Generate timestamp (from Reddit script)
    function getHumanReadableTimestamp() {
      const now = new Date();
      return `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-` +
             `${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}-` +
             `${String(now.getMinutes()).padStart(2,'0')}-${String(now.getSeconds()).padStart(2,'0')}`;
    }
  
    // Modified upload function with error handling
    async function uploadToGitHub(fileName, content) {
      const path = `reviews/${fileName}`;
      const apiUrl = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/contents/${path}`;
      
      try {
        const response = await fetch(apiUrl, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `token ${GITHUB_TOKEN}`
          },
          body: JSON.stringify({
            message: "Amazon review upload",
            content: base64EncodeUnicode(content),
            branch: GITHUB_BRANCH
          })
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(`GitHub API Error: ${errorData.message}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error("GitHub upload failed:", error);
        throw error;
      }
    }
  
    // ... [rest of the original Amazon script remains the same]
  
    // Modified display function with security check
    function displayOverallResult(data) {
      // ... [existing code]
      
      // Add GitHub upload button with security warning
      const githubButton = document.createElement("button");
      githubButton.innerText = "Upload to GitHub";
      githubButton.style.backgroundColor = "#2ea44f";
      
      githubButton.onclick = async () => {
        if (!GITHUB_TOKEN) {
          alert("GitHub credentials not configured! Check your .env file.");
          return;
        }
        
        try {
          await uploadToGitHub(`manual_upload_${getHumanReadableTimestamp()}.json`, JSON.stringify(results));
          alert("Successfully uploaded to GitHub!");
        } catch (error) {
          alert("Upload failed. Check console for details.");
        }
      };
      
      container.appendChild(githubButton);
    }
  })();