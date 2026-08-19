export const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'Hindi' },
  { code: 'bn', name: 'Bengali' }
];

export const getLanguageName = (code) => {
  const match = languages.find(l => l.code === code);
  return match ? match.name : code;
};
