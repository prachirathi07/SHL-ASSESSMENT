# SHL Assessment Recommender Frontend

A modern Next.js TypeScript frontend for the SHL Assessment Recommendation System.

## Features

- Built with Next.js 14 and TypeScript
- Responsive, modern UI with Tailwind CSS
- Form-based interface for entering queries or URLs
- API integration with the Python backend
- Beautiful card-based layout for assessment results

## Directory Structure

```
frontend/
├── app/                  # App Router directory
│   ├── api/              # API routes
│   │   └── recommend/    # Recommendation API endpoint
│   ├── about/            # About page
│   ├── globals.css       # Global styles
│   ├── layout.tsx        # Root layout component
│   └── page.tsx          # Home page component
├── components/           # Reusable UI components
├── public/               # Static assets
├── package.json          # Dependencies and scripts
├── tailwind.config.js    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Build for production:
   ```bash
   npm run build
   ```

## Integration with Backend

The frontend communicates with the Python backend through API routes. In a production environment, update the API routes to point to your actual backend URL.

## Customization

- Styles can be customized in `globals.css` and `tailwind.config.js`
- Component layouts are in their respective files under the `app` directory
- API integration can be adjusted in the `app/api/recommend/route.ts` file

## Deployment

The Next.js application can be deployed to various platforms like Vercel, Netlify, or as a static build on any web server.

```bash
# Build for production
npm run build

# Start the production server
npm start
``` 