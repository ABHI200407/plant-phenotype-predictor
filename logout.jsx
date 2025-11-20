import React from 'react'
import {Button} from '@mui/material';

export const logout = () => {
  const button={margin:'2rem',fontSize:'1.2rem',fontWeight:'700',backgroundColor:"blue",borderRadius:"0.5rem"};
  return (
    <Button style={button} variant="contained" color="error">logout</Button>
  )
}
 