"""Allow running: python -m services.agent.investigator NORAD_ID"""
from services.agent.investigator import main
raise SystemExit(main())
