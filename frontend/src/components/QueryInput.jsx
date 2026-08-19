import React, { useState } from 'react';
import { Search, Globe } from 'lucide-react';
import { languages } from '../utils/language';

const QueryInput = ({ onSubmit, disabled, selectedLang, setSelectedLang }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSubmit(query, selectedLang);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full relative z-10 isometric-container">
      <div className="floating-card p-2 flex items-center space-x-2">
        <div className="relative flex items-center">
          <Globe className="w-4 h-4 text-textMuted absolute left-3 pointer-events-none" />
          <select 
            value={selectedLang}
            onChange={(e) => setSelectedLang(e.target.value)}
            disabled={disabled}
            className="appearance-none bg-transparent text-sm font-medium text-text pl-9 pr-6 py-3 border-r border-border focus:outline-none focus:ring-2 focus:ring-primary/50 rounded-l-xl cursor-pointer"
          >
            <option value="auto">Auto (Detect)</option>
            {languages.map(l => (
              <option key={l.code} value={l.code} className="bg-background">
                {l.name}
              </option>
            ))}
          </select>
        </div>
        
        <input 
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
          placeholder="Ask HHG a question..."
          className="flex-1 bg-transparent border-none text-text focus:outline-none focus:ring-0 py-3 px-2 placeholder-textMuted/50 text-base"
        />
        
        <button 
          type="submit"
          disabled={disabled || !query.trim()}
          className="bg-primary hover:bg-primaryHover text-white p-3 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-glass focus:outline-none focus:ring-2 focus:ring-white/20"
          aria-label="Submit query"
        >
          <Search className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
};

export default QueryInput;
