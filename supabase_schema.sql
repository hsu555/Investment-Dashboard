create extension if not exists pgcrypto;

create table if not exists public.dashboard_users (
  id uuid primary key default gen_random_uuid(),
  username text not null unique,
  password_hash text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.holdings (
  id bigint generated always as identity primary key,
  user_id uuid not null references public.dashboard_users(id) on delete cascade,
  order_index integer not null,
  ticker text not null,
  quantity numeric not null default 0,
  purchase_price numeric not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists holdings_user_order_idx on public.holdings(user_id, order_index);

create table if not exists public.retirement_settings (
  user_id uuid primary key references public.dashboard_users(id) on delete cascade,
  current_age integer not null,
  retirement_age integer not null,
  life_expectancy integer not null,
  current_assets_wan numeric not null default 0,
  monthly_contribution_wan numeric not null default 0,
  monthly_expense_wan numeric not null default 0,
  mean_annual_return numeric not null default 0.07,
  annual_return_std numeric not null default 0.15,
  inflation_rate numeric not null default 0.03,
  n_simulations integer not null default 1000,
  updated_at timestamptz not null default now()
);

alter table public.dashboard_users enable row level security;
alter table public.holdings enable row level security;
alter table public.retirement_settings enable row level security;
