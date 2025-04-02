import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { 
  Box, 
  Button, 
  TextField, 
  Paper, 
  Typography, 
  List, 
  ListItem, 
  ListItemText,
  IconButton,
  CircularProgress,
  Divider,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Chip
} from '@mui/material';
import { Send, Upload, CloudUpload, Info } from '@mui/icons-material';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [modelType, setModelType] = useState('fast');
  const messagesEndRef = useRef(null);

  const modelDescriptions = {
    'fast': 'llama3-8b-8192 (Fast responses)',
    'powerful': 'llama3-70b-8192 (Higher quality)',
    'long-context': 'llama-3.3-70b-versatile (128K context)'
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/api/chat', {
        message: input,
        model_type: modelType
      }, {
        timeout: 15000 // 15 second timeout
      });

      const botMessage = { 
        text: response.data.response, 
        sender: 'bot',
        isMarkdown: true,
        model: response.data.model_used
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error('Detailed error:', {
        message: err.message,
        response: err.response?.data,
        stack: err.stack
      });
      
      let errorMsg = 'Failed to get response';
      if (err.response?.data?.error) {
        errorMsg = err.response.data.error;
      } else if (err.response?.data?.supported_models) {
        errorMsg = `Model error. Supported models: ${err.response.data.supported_models.join(', ')}`;
      } else if (err.message.includes('timeout')) {
        errorMsg = 'Request timed out. Please try again.';
      }
      
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const validTypes = ['application/pdf', 'text/plain', 
                         'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                         'text/markdown'];
      if (validTypes.includes(selectedFile.type)) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Invalid file type. Please upload PDF, TXT, DOCX, or MD files.');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      await axios.post('http://localhost:5000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(percentCompleted);
        },
      });

      setMessages(prev => [...prev, { 
        text: `Document "${file.name}" uploaded and processed successfully! You can now ask questions about it.`, 
        sender: 'bot' 
      }]);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.error || 'Failed to upload document');
    } finally {
      setUploading(false);
      setFile(null);
      setUploadProgress(0);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading) {
        handleSendMessage();
      }
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      height: '100vh',
      maxWidth: '800px',
      margin: '0 auto',
      padding: 2,
      backgroundColor: '#f5f5f5'
    }}>
      <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 2 }}>
        Groq Chatbot
      </Typography>
      
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Model Type</InputLabel>
        <Select
          value={modelType}
          label="Model Type"
          onChange={(e) => setModelType(e.target.value)}
          disabled={loading || uploading}
        >
          <MenuItem value="fast">{modelDescriptions.fast}</MenuItem>
          <MenuItem value="powerful">{modelDescriptions.powerful}</MenuItem>
          <MenuItem value="long-context">{modelDescriptions['long-context']}</MenuItem>
        </Select>
      </FormControl>

      <Paper elevation={3} sx={{ 
        flex: 1, 
        overflow: 'auto', 
        mb: 2,
        p: 2,
        backgroundColor: 'background.paper'
      }}>
        <List>
          {messages.map((msg, index) => (
            <React.Fragment key={index}>
              <ListItem 
                sx={{ 
                  justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  alignItems: 'flex-start'
                }}
              >
                <Paper
                  elevation={1}
                  sx={{
                    p: 2,
                    maxWidth: '70%',
                    backgroundColor: msg.sender === 'user' ? '#e3f2fd' : '#f1f1f1',
                    borderRadius: msg.sender === 'user' 
                      ? '18px 18px 0 18px' 
                      : '18px 18px 18px 0'
                  }}
                >
                  {msg.isMarkdown ? (
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  ) : (
                    <Typography variant="body1">{msg.text}</Typography>
                  )}
                  {msg.model && (
                    <Chip 
                      label={msg.model} 
                      size="small" 
                      sx={{ mt: 1, fontSize: '0.6rem' }}
                      icon={<Info fontSize="small" />}
                    />
                  )}
                </Paper>
              </ListItem>
              <Divider component="li" />
            </React.Fragment>
          ))}
          {loading && (
            <ListItem sx={{ justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </ListItem>
          )}
          <div ref={messagesEndRef} />
        </List>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <label htmlFor="file-upload">
          <input
            id="file-upload"
            type="file"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            accept=".pdf,.txt,.docx,.md"
            disabled={uploading || loading}
          />
          <IconButton 
            color="primary" 
            component="span"
            disabled={uploading || loading}
          >
            <Upload />
          </IconButton>
        </label>
        {file && (
          <>
            <Typography variant="body2" sx={{ flex: 1 }}>
              {file.name}
            </Typography>
            {uploading ? (
              <Box sx={{ width: '100%', mr: 1 }}>
                <LinearProgress variant="determinate" value={uploadProgress} />
                <Typography variant="caption" display="block" textAlign="center">
                  {uploadProgress}%
                </Typography>
              </Box>
            ) : (
              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={uploading || loading}
                startIcon={<CloudUpload />}
                size="small"
              >
                Upload
              </Button>
            )}
          </>
        )}
      </Box>

      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          fullWidth
          variant="outlined"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          multiline
          maxRows={4}
          disabled={loading || uploading}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleSendMessage}
          disabled={loading || uploading || !input.trim()}
          endIcon={<Send />}
        >
          Send
        </Button>
      </Box>
    </Box>
  );
};

export default Chatbot;