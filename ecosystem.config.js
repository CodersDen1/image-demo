module.exports = {
    apps: [
      {
        name: "frontend",
        script: "npm",
        args: "run dev -- --port 3333",
        cwd: "./frontend",
        watch: true,
        env: {
          NODE_ENV: "development",
        },
        env_production: {
          NODE_ENV: "production",
        },
      },
      {
        name: "backend",
        script: "python",
        args: "-m uvicorn main:app --host 0.0.0.0 --port 4444 --reload",
        cwd: "./backend",
        watch: true,
        env: {
          NODE_ENV: "development",
          PYTHONUNBUFFERED: "1"
        },
        env_production: {
          NODE_ENV: "production",
        },
      },
    ],
  };