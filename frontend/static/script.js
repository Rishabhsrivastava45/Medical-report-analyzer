document.addEventListener('DOMContentLoaded', () => {
    // --- AUTHENTICATION CHECK ---
    const token = localStorage.getItem('token');
    if (!token) {
        window.location.href = '/login';
        return;
    }

    const userName = localStorage.getItem('user_name') || 'User';
    const userNameDisplay = document.getElementById('user-name-display');
    const dropdownName = document.getElementById('dropdown-name');
    const userAvatar = document.getElementById('user-avatar');
    if(userNameDisplay) userNameDisplay.textContent = userName;
    if(dropdownName) dropdownName.textContent = userName;
    if(userAvatar) userAvatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(userName)}&background=2563eb&color=fff`;

    const logoutBtn = document.getElementById('logout-btn');
    if(logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            localStorage.removeItem('token');
            localStorage.removeItem('user_name');
            window.location.href = '/login';
        });
    }

    // --- THEME & DROPDOWN LOGIC ---
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.setAttribute('data-theme', 'dark');
        const themeSwitch = document.getElementById('theme-switch');
        if (themeSwitch) themeSwitch.checked = true;
    }

    const themeSwitch = document.getElementById('theme-switch');
    if (themeSwitch) {
        themeSwitch.addEventListener('change', (e) => {
            if (e.target.checked) {
                document.body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
            } else {
                document.body.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
            }
        });
    }

    const profileBtn = document.getElementById('profile-btn');
    const profileDropdown = document.getElementById('profile-dropdown');
    
    if (profileBtn && profileDropdown) {
        profileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            profileDropdown.classList.toggle('active');
        });

        document.addEventListener('click', (e) => {
            if (!profileDropdown.contains(e.target) && e.target !== profileBtn) {
                profileDropdown.classList.remove('active');
            }
        });
    }

    // --- ELEMENTS ---
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');
    const resultsView = document.getElementById('results-view');
    const analysisContent = document.getElementById('analysis-content');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat');
    
    let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    let isAnalyzed = false;

    // --- UPLOAD LOGIC ---
    if(uploadZone) {
        uploadZone.addEventListener('click', () => fileInput.click());

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.backgroundColor = '#dce7ff';
            uploadZone.style.borderColor = '#2563eb';
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.style.backgroundColor = '';
            uploadZone.style.borderColor = '';
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length) handleFileUpload(files[0]);
        });
    }

    if(fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) handleFileUpload(e.target.files[0]);
        });
    }

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);

        showLoader(true);
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Authorization': 'Bearer ' + token },
                body: formData
            });
            if (response.status === 401) { window.location.href = '/login'; return; }
            
            const data = await response.json();

            if (response.ok) {
                isAnalyzed = true;
                renderAnalysis(data.analysis);
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            console.error('Upload failed:', error);
            alert('An error occurred during analysis.');
        } finally {
            showLoader(false);
        }
    }

    function renderAnalysis(text) {
        if(!text) return;
        
        // 1. Process Risk Level for visual badge
        let processedText = text;
        const riskMatch = text.match(/\[RISK_LEVEL:\s*(LOW|MODERATE|HIGH)\]/i);
        
        if (riskMatch) {
            const level = riskMatch[1].toUpperCase();
            const badgeClass = `risk-${level.toLowerCase()}`;
            const riskHtml = `
                <div class="risk-level-container">
                    <span style="font-weight: 600; color: var(--text-muted);">Assessed Risk Level:</span>
                    <span class="risk-badge ${badgeClass}">${level}</span>
                </div>
            `;
            processedText = text.replace(/\[RISK_LEVEL:.*\]/i, riskHtml);
        }

        // 2. Use marked for rich markdown rendering
        analysisContent.innerHTML = marked.parse(processedText);
        
        // Highlight abnormal cells in tables
        const cells = analysisContent.querySelectorAll('td');
        cells.forEach(cell => {
            const text = cell.textContent.toLowerCase();
            if (text.includes('high') || text.includes('low') || text.includes('abnormal') || text.includes('reactive')) {
                cell.style.color = 'var(--danger)';
                cell.style.fontWeight = '700';
            }
        });
        
        // 3. UI feedback
        resultsView.classList.remove('hidden');
        
        // Smooth scroll to results with a small offset
        setTimeout(() => {
            const yOffset = -20; 
            const y = resultsView.getBoundingClientRect().top + window.pageYOffset + yOffset;
            window.scrollTo({top: y, behavior: 'smooth'});
        }, 100);
    }

    // --- CHAT LOGIC ---
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;
        if (!isAnalyzed) {
            addMessage('Please upload and analyze a report first.', 'bot');
            return;
        }

        addMessage(text, 'user');
        chatInput.value = '';
        
        const formData = new FormData();
        formData.append('message', text);
        formData.append('session_id', sessionId);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Authorization': 'Bearer ' + token },
                body: formData
            });
            if (response.status === 401) { window.location.href = '/login'; return; }

            const data = await response.json();
            
            if (response.ok) {
                addMessage(data.response, 'bot');
            } else {
                addMessage('Error: ' + data.error, 'bot');
            }
        } catch (error) {
            addMessage('Sorry, I lost connection to the server.', 'bot');
        }
    }

    function addMessage(text, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}-message`;
        if (type === 'bot') {
            msgDiv.innerHTML = marked.parse(text);
        } else {
            msgDiv.textContent = text;
        }
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    if(sendChatBtn) sendChatBtn.addEventListener('click', sendMessage);
    if(chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    // --- UTILS ---
    function showLoader(show) {
        if(loader) loader.classList.toggle('hidden', !show);
    }
    
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    if(newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', () => {
            switchView('dashboard');
            resultsView.classList.add('hidden');
            fileInput.value = '';
            uploadZone.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // --- NAVIGATION LOGIC ---
    const navLinks = document.querySelectorAll('.nav-link');
    const viewSections = document.querySelectorAll('.view-section');

    function switchView(viewId) {
        navLinks.forEach(link => {
            if (link.dataset.view === viewId) link.classList.add('active');
            else link.classList.remove('active');
        });

        viewSections.forEach(section => {
            if (section.id === `${viewId}-view`) section.classList.remove('hidden');
            else section.classList.add('hidden');
        });

        if (viewId === 'reports') {
            loadPastReports();
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchView(link.dataset.view);
        });
    });

    async function loadPastReports() {
        const grid = document.getElementById('reports-grid');
        grid.innerHTML = '<div class="loader-spinner" style="margin: 2rem auto; display: block;"></div>';
        try {
            const response = await fetch('/api/sessions', {
                headers: { 'Authorization': 'Bearer ' + token }
            });
            if (response.status === 401) { window.location.href = '/login'; return; }

            const data = await response.json();
            
            if (data.sessions && data.sessions.length > 0) {
                grid.innerHTML = '';
                data.sessions.forEach(session => {
                    const dateObj = new Date(session.updated_at);
                    const dateStr = dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    
                    const card = document.createElement('div');
                    card.className = 'card report-item';
                    card.innerHTML = `
                        <div class="report-item-header">
                            <h3><i class="fas fa-file-medical-alt"></i> Medical Report</h3>
                            <span style="font-size: 0.8rem; color: var(--text-muted);">${dateStr}</span>
                        </div>
                        <p class="snippet">${session.snippet}</p>
                        <button class="outline-btn" style="margin-top: 1rem; width: 100%;">View Analysis</button>
                    `;
                    card.addEventListener('click', () => {
                        loadSpecificReport(session.session_id);
                    });
                    grid.appendChild(card);
                });
            } else {
                grid.innerHTML = '<p style="color: var(--text-muted); text-align: center; grid-column: 1/-1;">No reports found. Upload your first report in the dashboard.</p>';
            }
        } catch (error) {
            console.error('Failed to load reports:', error);
            grid.innerHTML = '<p style="color: red; text-align: center; grid-column: 1/-1;">Failed to load reports. Please try again later.</p>';
        }
    }

    async function loadSpecificReport(id) {
        showLoader(true);
        try {
            const response = await fetch(`/api/report/${id}`, {
                headers: { 'Authorization': 'Bearer ' + token }
            });
            if (response.status === 401) { window.location.href = '/login'; return; }

            const data = await response.json();
            if (response.ok) {
                sessionId = id; 
                isAnalyzed = true;
                switchView('dashboard');
                renderAnalysis(data.analysis);
            } else {
                alert('Error loading report: ' + data.detail);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Failed to load the specific report.');
        } finally {
            showLoader(false);
        }
    }
});

// --- CLIENT-SIDE PDF DOWNLOAD ---
window.triggerServerDownload = function() {
    console.log('Triggering client-side PDF download...');
    const element = document.getElementById('analysis-content');
    
    if (!element || element.innerHTML.trim() === '') {
        alert('No analysis available to download.');
        return;
    }

    const btn = document.getElementById('print-btn');
    const originalContent = btn.innerHTML;
    
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating PDF...';
    btn.style.pointerEvents = 'none';

    const opt = {
        margin:       15,
        filename:     `Medical_Report_Analysis.pdf`,
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, useCORS: true, letterRendering: true },
        jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };

    html2pdf().set(opt).from(element).save().then(() => {
        btn.innerHTML = originalContent;
        btn.style.pointerEvents = 'auto';
    }).catch(err => {
        console.error("PDF generation failed:", err);
        btn.innerHTML = originalContent;
        btn.style.pointerEvents = 'auto';
        alert('Failed to generate PDF. Please try again.');
    });
};
