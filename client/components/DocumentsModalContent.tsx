
import React from 'react';
import { DocumentItem } from '../types';

import { DocumentTextIcon, ArrowDownTrayIcon, EyeIcon, PaperAirplaneIcon } from './icons';

interface DocumentsModalContentProps {
  documents: DocumentItem[];
  T: any; // Translation object
}

const getFileIcon = (): React.ReactNode => {
  return <DocumentTextIcon className="w-8 h-8 text-accent" />; 
};

const DocumentsModalContent: React.FC<DocumentsModalContentProps> = ({ documents, T }) => {
  const labels = T.documentsContent || {};
  const documentTypeNames = labels.documentTypes || {
    'PDF': "PDF Document",
    'DOCX': "Word Document",
    'TXT': "Text File",
    'Email': "Email Message",
    'Image': "Image File",
  };


  const handlePreview = (doc: DocumentItem) => {
    if (doc.url && doc.url !== '#') {
      window.open(doc.url, '_blank');
    } else {
      alert(labels.previewNotAvailable || "Preview not available for this document.");
    }
  };

  const handleDownload = (doc: DocumentItem) => {
     if (doc.url && doc.url !== '#') {
      const link = document.createElement('a');
      link.href = doc.url;
      link.setAttribute('download', doc.name);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      alert(labels.downloadNotAvailable || "Download not available for this document.");
    }
  };

  const handleSend = (doc: DocumentItem) => {
    const messageTemplate = labels.simulatingSend || "Simulating send for: {docName}";
    const message = messageTemplate.replace('{docName}', doc.name);
    alert(message);
  };

  if (documents.length === 0) {
    return <p className="text-text-secondary">{labels.noDocumentsFound || "No documents found."}</p>;
  }

  return (
    <div className="flex-grow overflow-y-auto bg-[#f4f0ff] p-4 sm:p-6 rounded-b-lg">
      <div className="space-y-4">
        {documents.map((doc) => (
          <div key={doc.id} className="bg-[#f8f5fc] p-4 rounded-lg shadow-sm border border-border-color flex items-center gap-4 flex-wrap justify-between w-full min-w-0">
            <div className="flex items-center gap-3 min-w-0 w-full">
              {getFileIcon()}
              <div className="flex flex-col gap-1 min-w-0 w-full">
                <p className="text-sm font-semibold text-text-primary break-words max-w-[220px]" title={doc.name}>{doc.name}</p>
                <p className="text-xs text-text-secondary leading-tight break-words max-w-full">
                  Type: {documentTypeNames[doc.type] || doc.type} • Added: {doc.dateAdded} • Size: {doc.size}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                className="rounded-full bg-accent/10 p-2 text-accent hover:bg-accent/20 focus:outline-none focus:ring-2 focus:ring-accent"
                onClick={() => handlePreview(doc)}
                type="button"
                title="Preview"
                aria-label={`Preview ${doc.name}`}
              >
                <EyeIcon className="w-5 h-5" />
              </button>
              <button
                className={`rounded-full bg-green-100 p-2 text-green-700 hover:bg-green-200 focus:outline-none focus:ring-2 focus:ring-green-400 transition ${!doc.url || doc.url === '#' ? 'opacity-60 cursor-not-allowed' : ''}`}
                onClick={() => { if (doc.url && doc.url !== '#') handleDownload(doc); }}
                type="button"
                title={doc.url && doc.url !== '#' ? 'Download' : 'Download not available'}
                aria-label={`Download ${doc.name}`}
                disabled={!doc.url || doc.url === '#'}
              >
                <ArrowDownTrayIcon className="w-5 h-5" />
                <span className="hidden sm:inline font-semibold text-xs">Download</span>
                <span className="sm:hidden block text-[10px] font-semibold ml-1">Download</span>
              </button>
              <button
                className="rounded-full bg-blue-100 p-2 text-blue-700 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400"
                onClick={() => handleSend(doc)}
                type="button"
                title="Send"
                aria-label={`Send ${doc.name}`}
              >
                <PaperAirplaneIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DocumentsModalContent;