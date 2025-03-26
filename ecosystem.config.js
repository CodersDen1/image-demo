module.exports = {
    apps: [
      {
        name: "frontend",
        script: "npm",
        args: "run start",
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
        args:  "-c 'source myenv/bin/activate && python -m uvicorn main:app --host 0.0.0.0 --port 3000 --reload'",
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