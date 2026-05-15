#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:35:33 2026

@author: marcozini
"""

## Implement formulas of the most Greeks connected to Black-Scholes

# Delta
# Vega
# Theta
# Roh
# Gamma


# Preliminiaries

import numpy as np
from scipy import stats


def d1(S, K, r, T, sigma, q = 0):
    d1 = (np.log(S/K) + (r - q + sigma**2 / 2)*T) / (sigma * np.sqrt(T))
    return d1

def d2(S, K, r, T, sigma, q = 0):
    d2 = (np.log(S/K) + (r - q + sigma**2 / 2)*T) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T)
    return d2    


#Validation function --> create later a seperate file liker helpers_quant or validate.py ....

def validate_bs_inputs(S, K, r, T, sigma, q, option_type):
    
    # Input validation
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if q < 0:
        raise ValueError("Dividend yield can not be negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")


## Greeks

# Delta: dV/dS

def BS_delta(S, K, r, T, sigma, q = 0, option_type = "call"):
    
    # Input validation
    validate_bs_inputs(S, K, r, T, sigma, q, option_type)
  
    # Calculate d1
    
    d_1 = d1(S, K, r, T, sigma, q)
    
    # Call option
    if option_type == "call":
        delta = np.exp(-q*T) * stats.norm.cdf(d_1)
    
    # Put option
    elif option_type == "put":
        delta = -np.exp(-q*T) * stats.norm.cdf(-d_1)
    
    return delta



# Vega: dV/dsigma
def BS_vega(S, K, r, T, sigma, q = 0, option_type = "call"):
    
    # Input validation
    validate_bs_inputs(S, K, r, T, sigma, q, option_type)
    
    # Calculate d_1
    d_1 = d1(S, K, r, T, sigma, q)
    
    # Call and put option
    if option_type in ["call", "put"]:
        vega = S*np.exp(-q*T) * stats.norm.pdf(d_1) * np.sqrt(T)
    
    return vega

    
# Theta: dV/dtau
def BS_theta(S, K, r, T, sigma, q = 0, option_type = "call"):
    
    # Input validation
    validate_bs_inputs(S, K, r, T, sigma, q, option_type)
    
    #Calculate d1 & d2
    d_1 = d1(S, K, r, T, sigma, q)
    d_2 = d2(S, K, r, T, sigma, q)
    
    # Call option
    if option_type == "call":
        theta = (
            -np.exp(-q*T) * S * stats.norm.pdf(d_1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r*T) * stats.norm.cdf(d_2)
            + q * S * np.exp(-q*T) * stats.norm.cdf(d_1)
        )
    
    # Put option
    elif option_type == "put":
        theta = (
            - np.exp(-q*T) * S * stats.norm.pdf(d_1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r*T) * stats.norm.cdf(-d_2)
            - q * S * np.exp(-q*T) * stats.norm.cdf(-d_1)
        )
    
    return theta


# Roh: dv/dr
def BS_rho(S, K, r, T, sigma, q = 0, option_type = "call"):
    
    # Input validation
    validate_bs_inputs(S, K, r, T, sigma, q, option_type)
    
    # Calculate d2
    d_2 = d2(S, K, r, T, sigma, q)
    
    # Call option
    if option_type == "call":
        rho = K * T * np.exp(-r*T) * stats.norm.cdf(d_2)
        
    # Put option
    elif option_type == "put":
        rho = -K * T * np.exp(-r*T) * stats.norm.cdf(-d_2)
        
    return rho


# Gamma: ddelta/dS = d^2V/dS^2
def BS_gamma(S, K, r, T, sigma, q = 0, option_type = "call"):
    
    # Input validation
    validate_bs_inputs(S, K, r, T, sigma, q, option_type)
    
    # Calculate d1
    d_1 = d1(S, K, r, T, sigma, q)

    # Call and put option
    if option_type in ["call", "put"]:
        gamma = np.exp(-q*T) * stats.norm.pdf(d_1) / (S * sigma * np.sqrt(T))
    
    return gamma

    

## TODO Checks --> Look better 

S, K, r, T, sigma = 100, 100, 0.01, 1, 0.2

# Delta ATM call ~ 0.5
print(BS_delta(S, K, r, T, sigma, option_type="call"))

# Delta call + |Delta put| should = 1
print(BS_delta(S, K, r, T, sigma, option_type="call") + 
      abs(BS_delta(S, K, r, T, sigma, option_type="put")))

# Gamma identical for call and put
print(BS_gamma(S, K, r, T, sigma, option_type="call"))
print(BS_gamma(S, K, r, T, sigma, option_type="put"))

# Vega identical for call and put
print(BS_vega(S, K, r, T, sigma, option_type="call"))
print(BS_vega(S, K, r, T, sigma, option_type="put"))




