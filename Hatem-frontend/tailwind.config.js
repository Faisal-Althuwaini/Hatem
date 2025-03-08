/** @type {import('tailwindcss').Config} */
export default {
  experimental: {
    optimizeUniversalDefaults: true, // Improves compatibility
  },
      darkMode: "class", // Forces dark mode via class instead of media query
    content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
      extend: {
        colors: {
          colors: {
            background: "oklch(1 0 0)", // Default for modern browsers
            primary: "oklch(0.21 0.006 285.885)",
            secondary: "oklch(0.967 0.001 286.375)",
            destructive: "oklch(0.577 0.245 27.325)",
            ring: "oklch(0.705 0.015 286.067)",
          },       
          cyan: {
            50: "rgb(236, 254, 255)", 
            100: "rgb(207, 250, 254)", 
            200: "rgb(165, 243, 252)", 
            300: "rgb(103, 232, 249)", 
            400: "rgb(34, 211, 238)", 
            500: "rgb(6, 182, 212)", 
            600: "rgb(8, 145, 178)", // Instead of `oklch()`
            700: "rgb(14, 116, 144)", 
            800: "rgb(21, 94, 117)", 
            900: "rgb(22, 78, 99)", 
            950: "rgb(8, 51, 68)", 
          },
  
        },
    
      },
    },
    plugins: [],
  };
  