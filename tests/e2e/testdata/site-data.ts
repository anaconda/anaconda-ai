// Base URL from .env (used for CLI commands e.g. sites modify --domain)
export const BASE_URL = process.env.BASE_URL ?? 'https://qa.anaconda-sandbox.com';
export const baseDomain = new URL(BASE_URL).hostname;

/** Self-hosted site name used with `anaconda sites list|add|modify`. */
export const SELF_HOSTED_SITE_NAME = 'self-hosted';
