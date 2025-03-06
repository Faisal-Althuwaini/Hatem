/** @type {import('tailwindcss').Config} */
export default {
    darkMode: "class", // Forces dark mode via class instead of media query
    content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
      extend: {
        colors: {
            primary: 'rgb(21, 94, 117)', // Equivalent to Tailwind's cyan-800
          },
    
      },
    },
    plugins: [],
  };
  